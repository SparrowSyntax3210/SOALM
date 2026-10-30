import json
import math
import time
from pathlib import Path
from datetime import datetime, timezone

import requests
from shapely.geometry import Point, shape

# ============================================================
# SANKET WEATHER CACHE
# ============================================================

GRID_SIZE_KM = 20

# Required historical points
HISTORY_HOURS = [0, 3, 6, 12, 24, 48, 72]

# Number of coordinates sent in one Open-Meteo request
BATCH_SIZE = 100

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

TIMEOUT = 120

BASE_DIR = Path(__file__).resolve().parent

BOUNDARY_FILE = BASE_DIR / "india.geojson"
OUTPUT_FILE = BASE_DIR / "weather.json"


# ============================================================
# WEATHER VARIABLES REQUIRED BY SOALM
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
# OPEN-METEO PARAMETERS
# ============================================================


def weather_params():

    return {
        "hourly": ",".join(HOURLY_VARIABLES),
        # We need 72 hours of history
        "past_days": 3,
        # Not required for the cache currently
        "forecast_days": 0,
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }


# ============================================================
# LOAD INDIA BOUNDARY
# ============================================================


def load_india_boundary():

    print("Loading India boundary...")

    with open(BOUNDARY_FILE, "r", encoding="utf-8") as file:

        geojson = json.load(file)

    boundaries = []

    if geojson["type"] == "FeatureCollection":

        for feature in geojson["features"]:

            if feature.get("geometry"):

                boundaries.append(shape(feature["geometry"]))

    elif geojson["type"] == "Feature":

        boundaries.append(shape(geojson["geometry"]))

    else:

        boundaries.append(shape(geojson))

    return boundaries


# ============================================================
# GENERATE INDIA 20 KM GRID
# ============================================================


def generate_india_grid(boundaries):

    print("Generating 20 km India grid...")

    min_lat = 8.0
    max_lat = 37.0

    min_lon = 68.0
    max_lon = 97.5

    lat_step = GRID_SIZE_KM / 111.0

    grid = []

    grid_id = 0

    lat = min_lat

    while lat <= max_lat:

        lon_step = GRID_SIZE_KM / (111.0 * math.cos(math.radians(lat)))

        lon = min_lon

        while lon <= max_lon:

            point = Point(lon, lat)

            # covers() includes boundary points
            inside = any(boundary.covers(point) for boundary in boundaries)

            if inside:

                grid_id += 1

                grid.append(
                    {
                        "gridId": f"IND_{grid_id:06d}",
                        "latitude": round(lat, 5),
                        "longitude": round(lon, 5),
                    }
                )

            lon += lon_step

        lat += lat_step

    print(f"Generated {len(grid)} weather grids.")

    return grid


# ============================================================
# FETCH BATCH
# ============================================================


def fetch_weather_batch(points):

    if not points:
        return []

    latitudes = ",".join(str(float(p["latitude"])) for p in points)

    longitudes = ",".join(str(float(p["longitude"])) for p in points)

    params = weather_params()

    params["latitude"] = latitudes
    params["longitude"] = longitudes

    print(f"Fetching {len(points)} grid points...")

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=TIMEOUT,
        headers={"User-Agent": "SOALM-DisasterOS/2.0", "Accept": "application/json"},
    )

    response.raise_for_status()

    data = response.json()

    if len(points) == 1:

        if not isinstance(data, dict):

            raise RuntimeError("Invalid Open-Meteo response")

        return [data]

    if not isinstance(data, list):

        raise RuntimeError(
            "Open-Meteo did not return " "a list for multiple coordinates"
        )

    if len(data) != len(points):

        raise RuntimeError(
            f"Expected {len(points)} responses " f"but received {len(data)}"
        )

    return data


# ============================================================
# FIND HOURLY INDEX
# ============================================================


def find_index(times, hours_ago):

    if hours_ago == 0:

        return len(times) - 1

    target_index = len(times) - 1 - hours_ago

    if target_index < 0:

        return None

    return target_index


# ============================================================
# EXTRACT WEATHER POINT
# ============================================================


def extract_weather_point(hourly, index):

    result = {}

    for variable in HOURLY_VARIABLES:

        values = hourly.get(variable, [])

        if index is None or index >= len(values):

            result[variable] = 0.0

        else:

            value = values[index]

            if value is None:

                value = 0.0

            result[variable] = value

    return result


# ============================================================
# CALCULATE RAINFALL WINDOWS
# ============================================================


def calculate_rainfall(hourly, current_index):

    rain = hourly.get("rain", [])

    def total(hours):

        start = max(0, current_index - hours + 1)

        values = rain[start : current_index + 1]

        return round(sum(float(v or 0) for v in values), 3)

    return {
        "rain_last_3h": total(3),
        "rain_last_6h": total(6),
        "rain_last_12h": total(12),
        "rain_last_24h": total(24),
        "rain_last_48h": total(48),
        "rain_last_72h": total(72),
    }


# ============================================================
# PROCESS ONE GRID
# ============================================================


def process_grid(grid, response):

    hourly = response.get("hourly")

    if not isinstance(hourly, dict):

        raise RuntimeError("Missing hourly data")

    times = hourly.get("time", [])

    if not times:

        raise RuntimeError("No hourly timestamps")

    current_index = len(times) - 1

    weather = {}

    # --------------------------------------------------------
    # Historical snapshots
    # --------------------------------------------------------

    for hours_ago in HISTORY_HOURS:

        index = find_index(times, hours_ago)

        if hours_ago == 0:

            key = "current"

        else:

            key = f"{hours_ago}h"

        weather[key] = {
            "time": (times[index] if index is not None else None),
            "data": extract_weather_point(hourly, index),
        }

    # --------------------------------------------------------
    # Rainfall accumulation
    # --------------------------------------------------------

    rainfall = calculate_rainfall(hourly, current_index)

    weather["rainfall"] = rainfall

    # --------------------------------------------------------
    # Final grid object
    # --------------------------------------------------------

    return {
        "gridId": grid["gridId"],
        "latitude": grid["latitude"],
        "longitude": grid["longitude"],
        "weather": weather,
    }


# ============================================================
# GENERATE WEATHER JSON
# ============================================================


def generate_weather_json():

    print()
    print("=" * 50)
    print(" SANKET WEATHER CACHE GENERATOR")
    print("=" * 50)
    print()

    # --------------------------------------------------------
    # Boundary
    # --------------------------------------------------------

    boundaries = load_india_boundary()

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    grids = generate_india_grid(boundaries)

    # --------------------------------------------------------
    # JSON structure
    # --------------------------------------------------------

    weather_cache = {
        "metadata": {
            "application": "Sanket",
            "source": "Open-Meteo",
            "gridSizeKm": GRID_SIZE_KM,
            "historyHours": HISTORY_HOURS,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "totalGrids": len(grids),
            "variables": HOURLY_VARIABLES,
        },
        "grids": {},
    }

    # --------------------------------------------------------
    # Process batches
    # --------------------------------------------------------

    total = len(grids)

    total_batches = math.ceil(total / BATCH_SIZE)

    for start in range(0, total, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total)

        batch = grids[start:end]

        batch_number = (start // BATCH_SIZE) + 1

        print()
        print(f"Batch {batch_number}/" f"{total_batches}")

        try:

            responses = fetch_weather_batch(batch)

            for grid, response in zip(batch, responses):

                try:

                    processed = process_grid(grid, response)

                    weather_cache["grids"][grid["gridId"]] = processed

                except Exception as error:

                    print(f"Grid " f"{grid['gridId']} failed: " f"{error}")

        except Exception as error:

            print(f"Batch failed: {error}")

        # Avoid hammering API
        time.sleep(1)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print()
    print("Writing weather.json...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(weather_cache, file, indent=2, ensure_ascii=False)

    print()
    print("=" * 50)
    print(" weather.json CREATED")
    print("=" * 50)

    print(f"File: {OUTPUT_FILE}")

    print(f"Grids: " f"{len(weather_cache['grids'])}")

    print("Historical snapshots: " "0h, 3h, 6h, 12h, 24h, 48h, 72h")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    generate_weather_json()
