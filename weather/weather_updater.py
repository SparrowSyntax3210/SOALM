from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import requests

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

WEATHER_FILE = BASE_DIR / "weather.json"


# ============================================================
# CONFIGURATION
# ============================================================

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

REQUEST_TIMEOUT = 30

CLOCK_CHECK_SECONDS = 30


# ============================================================
# OPEN-METEO VARIABLES
# ============================================================

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "showers",
    "cloud_cover",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
]


# ============================================================
# GLOBAL THREAD STATE
# ============================================================

_weather_updater_thread: threading.Thread | None = None

_weather_updater_stop_event = threading.Event()


# ============================================================
# LOGGING
# ============================================================


def log(message: str) -> None:
    """
    Print a timestamped updater message.
    """

    timestamp = datetime.now().strftime("%H:%M:%S")

    print(
        f"[WEATHER-UPDATER] {timestamp} | {message}",
        flush=True,
    )


# ============================================================
# CACHE LOAD
# ============================================================


def load_cache() -> dict[str, Any]:
    """
    Load weather.json.

    Expected structure:

    {
        "metadata": {...},
        "grids": {
            "grid_xxx": {
                "gridId": "...",
                "latitude": 28.123,
                "longitude": 77.123,
                "weather": {...}
            }
        }
    }
    """

    if not WEATHER_FILE.exists():

        raise FileNotFoundError(f"weather.json not found: {WEATHER_FILE}")

    with WEATHER_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        cache = json.load(file)

    if not isinstance(cache, dict):

        raise RuntimeError("weather.json root must be an object")

    grids = cache.get("grids")

    if not isinstance(grids, dict):

        raise RuntimeError("weather.json must contain a 'grids' object")

    return cache


# ============================================================
# ATOMIC CACHE SAVE
# ============================================================


def save_cache_atomic(
    cache: dict[str, Any],
) -> None:
    """
    Safely replace weather.json.

    A temporary file is written first so that an interrupted
    write does not corrupt the main cache.
    """

    temp_file = WEATHER_FILE.with_suffix(".tmp")

    with temp_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            cache,
            file,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    temp_file.replace(WEATHER_FILE)


# ============================================================
# UTC HOUR IDENTIFIER
# ============================================================


def current_utc_hour() -> str:
    """
    Returns the current UTC hour.

    Example:

        2026-09-09T05
    """

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")


# ============================================================
# WAIT UNTIL NEXT HOUR
# ============================================================


def seconds_until_next_hour() -> int:
    """
    Calculate seconds until the next UTC hour.
    """

    now = datetime.now(timezone.utc)

    next_hour = now.replace(
        minute=0,
        second=0,
        microsecond=0,
    ) + timedelta(hours=1)

    seconds = (next_hour - now).total_seconds()

    return max(
        1,
        int(seconds),
    )


# ============================================================
# INTERNET CHECK
# ============================================================


def internet_available() -> bool:
    """
    Lightweight connectivity check.

    IMPORTANT:

    This is only a connectivity check.

    It does NOT download weather data.
    """

    try:

        response = requests.get(
            "https://api.open-meteo.com",
            timeout=5,
            headers={"User-Agent": "Sanket-SOALM-WeatherCache/1.0"},
        )

        return response.ok

    except requests.RequestException:

        return False


# ============================================================
# FETCH WEATHER FOR ONE LOCATION
# ============================================================


def fetch_weather(
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    """
    Fetch weather for ONE coordinate.

    This function is intentionally location-specific.

    Example:

        fetch_weather(
            28.6139,
            77.2090
        )

    results in ONE Open-Meteo request.

    It does NOT request all India grids.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        # We need historical data because SOALM uses
        # 3h / 6h / 12h / 24h / 48h / 72h features.
        "past_days": 3,
        # No forecast required for the offline cache.
        "forecast_days": 0,
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={
            "User-Agent": "Sanket-SOALM-WeatherCache/1.0",
            "Accept": "application/json",
        },
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, dict):

        raise RuntimeError("Open-Meteo returned invalid data")

    if not isinstance(
        data.get("hourly"),
        dict,
    ):

        raise RuntimeError("Open-Meteo response has no hourly data")

    return data


# ============================================================
# FIND NEAREST GRID
# ============================================================


def find_nearest_grid(
    latitude: float,
    longitude: float,
    cache: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """
    Find the nearest 20-km cached grid.

    This is used only when we explicitly want to update
    a particular requested location.

    It does NOT call Open-Meteo.
    """

    grids = cache.get(
        "grids",
        {},
    )

    nearest_id: str | None = None

    nearest_grid: dict[str, Any] | None = None

    nearest_distance = float("inf")

    lat_scale = math.cos(math.radians(latitude))

    for grid_id, grid in grids.items():

        if not isinstance(
            grid,
            dict,
        ):
            continue

        try:

            grid_lat = float(grid["latitude"])

            grid_lon = float(grid["longitude"])

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            continue

        d_lat = grid_lat - latitude

        d_lon = (grid_lon - longitude) * lat_scale

        distance = d_lat * d_lat + d_lon * d_lon

        if distance < nearest_distance:

            nearest_distance = distance

            nearest_id = grid_id

            nearest_grid = grid

    if nearest_id is None or nearest_grid is None:

        raise RuntimeError("Could not find nearest cached grid")

    return (
        nearest_id,
        nearest_grid,
    )


# ============================================================
# UPDATE ONE GRID
# ============================================================


def update_grid(
    grid: dict[str, Any],
    weather: dict[str, Any],
) -> None:
    """
    Store compact weather history in one grid.

    This function expects a normal Open-Meteo hourly response.
    """

    hourly = weather.get("hourly")

    if not isinstance(
        hourly,
        dict,
    ):

        raise RuntimeError("Weather response contains no hourly data")

    times = hourly.get(
        "time",
        [],
    )

    if (
        not isinstance(
            times,
            list,
        )
        or not times
    ):

        raise RuntimeError("Weather response contains no timestamps")

    # --------------------------------------------------------
    # Locate the latest available observation
    # --------------------------------------------------------

    current_index = len(times) - 1

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def value(
        variable: str,
        index: int,
    ) -> float:

        values = hourly.get(
            variable,
            [],
        )

        if not isinstance(
            values,
            list,
        ):

            return 0.0

        if index < 0 or index >= len(values):

            return 0.0

        raw = values[index]

        if raw is None:

            return 0.0

        try:

            number = float(raw)

            if math.isfinite(number):

                return number

        except (
            TypeError,
            ValueError,
        ):

            pass

        return 0.0

    # --------------------------------------------------------
    # Compact snapshots
    #
    # We keep exactly what SOALM needs:
    #
    # current
    # 3h
    # 6h
    # 12h
    # 24h
    # 48h
    # 72h
    # --------------------------------------------------------

    snapshots: dict[str, Any] = {}

    snapshot_hours = [
        0,
        3,
        6,
        12,
        24,
        48,
        72,
    ]

    for hours_ago in snapshot_hours:

        index = current_index - hours_ago

        if index < 0:

            index = 0

        key = "current" if hours_ago == 0 else f"{hours_ago}h"

        data: dict[str, Any] = {}

        for variable in HOURLY_VARIABLES:

            data[variable] = value(
                variable,
                index,
            )

        snapshots[key] = {
            "time": times[index],
            "data": data,
        }

    # --------------------------------------------------------
    # Rainfall totals
    # --------------------------------------------------------

    rain = hourly.get(
        "rain",
        [],
    )

    if not isinstance(
        rain,
        list,
    ):

        rain = []

    def rain_sum(
        hours: int,
    ) -> float:

        start = max(
            0,
            current_index - hours + 1,
        )

        total = 0.0

        for index in range(
            start,
            current_index + 1,
        ):

            total += value(
                "rain",
                index,
            )

        return round(
            total,
            3,
        )

    current_pressure = value(
        "surface_pressure",
        current_index,
    )

    current_temperature = value(
        "temperature_2m",
        current_index,
    )

    current_humidity = value(
        "relative_humidity_2m",
        current_index,
    )

    current_wind = value(
        "wind_speed_10m",
        current_index,
    )

    # --------------------------------------------------------
    # Feature deltas
    # --------------------------------------------------------

    rainfall = {
        "rain_last_1h": rain_sum(1),
        "rain_last_3h": rain_sum(3),
        "rain_last_6h": rain_sum(6),
        "rain_last_12h": rain_sum(12),
        "rain_last_24h": rain_sum(24),
        "rain_last_48h": rain_sum(48),
        "rain_last_72h": rain_sum(72),
        "pressure_change_1h": current_pressure
        - value(
            "surface_pressure",
            current_index - 1,
        ),
        "pressure_change_3h": current_pressure
        - value(
            "surface_pressure",
            current_index - 3,
        ),
        "pressure_change_6h": current_pressure
        - value(
            "surface_pressure",
            current_index - 6,
        ),
        "temperature_change_3h": current_temperature
        - value(
            "temperature_2m",
            current_index - 3,
        ),
        "temperature_change_6h": current_temperature
        - value(
            "temperature_2m",
            current_index - 6,
        ),
        "humidity_change_3h": current_humidity
        - value(
            "relative_humidity_2m",
            current_index - 3,
        ),
        "humidity_change_6h": current_humidity
        - value(
            "relative_humidity_2m",
            current_index - 6,
        ),
        "wind_change_3h": current_wind
        - value(
            "wind_speed_10m",
            current_index - 3,
        ),
        "wind_change_6h": current_wind
        - value(
            "wind_speed_10m",
            current_index - 6,
        ),
    }

    # --------------------------------------------------------
    # Final cached weather object
    # --------------------------------------------------------

    grid["weather"] = {
        **snapshots,
        "rainfall": rainfall,
    }

    grid["weatherUpdatedAt"] = datetime.now(timezone.utc).isoformat()


# ============================================================
# UPDATE SPECIFIC LOCATION
# ============================================================


def update_location(
    latitude: float,
    longitude: float,
) -> bool:
    """
    Update ONLY the nearest grid for the supplied location.

    IMPORTANT:

    This function makes ONE Open-Meteo request.

    It does NOT update all 2600 grids.
    """

    cache = load_cache()

    grid_id, grid = find_nearest_grid(
        latitude,
        longitude,
        cache,
    )

    log(f"Updating requested location " f"{latitude:.4f}, {longitude:.4f}")

    log(f"Nearest cache grid: {grid_id}")

    # --------------------------------------------------------
    # ONE API REQUEST
    # --------------------------------------------------------

    weather = fetch_weather(
        latitude,
        longitude,
    )

    # --------------------------------------------------------
    # Update only this grid
    # --------------------------------------------------------

    update_grid(
        grid,
        weather,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = cache.setdefault(
        "metadata",
        {},
    )

    now = datetime.now(timezone.utc)

    metadata["updatedAt"] = now.isoformat()

    metadata["lastUpdateMode"] = "single-location"

    metadata["lastUpdateLatitude"] = latitude

    metadata["lastUpdateLongitude"] = longitude

    metadata["lastUpdatedGrid"] = grid_id

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_cache_atomic(cache)

    log(f"Grid {grid_id} updated successfully.")

    return True


# ============================================================
# HOURLY CACHE HOOK
# ============================================================


def hourly_update() -> None:
    """
    Called once when a NEW UTC hour begins.

    IMPORTANT:

    We deliberately DO NOT refresh all India grids here.

    Live weather is fetched by SOALM only for locations
    actually requested by users.

    weather.json remains the offline fallback cache.
    """

    log("New UTC hour detected.")

    log("No nationwide Open-Meteo refresh performed.")

    log("weather.json remains the offline spatial cache.")

    log("Live weather remains location-specific.")


# ============================================================
# BACKGROUND LOOP
# ============================================================


def _updater_loop() -> None:
    """
    Background daemon loop.

    It waits for the next UTC hour and then calls
    hourly_update() exactly once.

    It does NOT call Open-Meteo every minute.
    """

    last_hour = current_utc_hour()

    log(f"Updater clock initialized: {last_hour} UTC")

    while not _weather_updater_stop_event.is_set():

        try:

            current_hour = current_utc_hour()

            # ------------------------------------------------
            # New hour detected
            # ------------------------------------------------

            if current_hour != last_hour:

                last_hour = current_hour

                hourly_update()

            # ------------------------------------------------
            # Sleep
            # ------------------------------------------------

            _weather_updater_stop_event.wait(CLOCK_CHECK_SECONDS)

        except Exception as exc:

            log(f"Background updater error: {exc}")

            _weather_updater_stop_event.wait(CLOCK_CHECK_SECONDS)


# ============================================================
# START BACKGROUND UPDATER
# ============================================================


def start_weather_updater() -> None:
    """
    Start the background updater.

    Safe to call multiple times.

    Only one updater thread will run.
    """

    global _weather_updater_thread

    # --------------------------------------------------------
    # Already running
    # --------------------------------------------------------

    if _weather_updater_thread is not None and _weather_updater_thread.is_alive():

        log("Background updater already running.")

        return

    # --------------------------------------------------------
    # Reset stop event
    # --------------------------------------------------------

    _weather_updater_stop_event.clear()

    # --------------------------------------------------------
    # Create daemon thread
    # --------------------------------------------------------

    _weather_updater_thread = threading.Thread(
        target=_updater_loop,
        name="SanketWeatherUpdater",
        daemon=True,
    )

    _weather_updater_thread.start()

    log("Background weather updater started.")

    log("Updater will wait for the next UTC hour.")


# ============================================================
# STOP BACKGROUND UPDATER
# ============================================================


def stop_weather_updater() -> None:
    """
    Stop the background updater gracefully.
    """

    global _weather_updater_thread

    if _weather_updater_thread is None:

        return

    log("Stopping weather updater...")

    _weather_updater_stop_event.set()

    _weather_updater_thread.join(timeout=5)

    _weather_updater_thread = None

    log("Weather updater stopped.")


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    log("=" * 64)

    log("SANKET / SOALM WEATHER CACHE UPDATER")

    log(f"Cache: {WEATHER_FILE}")

    log("Mode: LOCATION-SPECIFIC")

    log("Nationwide refresh: DISABLED")

    log("Hourly background hook: ENABLED")

    log("=" * 64)

    start_weather_updater()

    try:

        while True:

            time.sleep(3600)

    except KeyboardInterrupt:

        log("Keyboard interrupt received.")

        stop_weather_updater()
