from __future__ import annotations

import os
import sys
import json
import math
import time
import shutil
import tempfile
import traceback
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = PROJECT_ROOT / "data" / "era5"

RAW_ROOT = OUTPUT_ROOT / "raw"
FEATURE_ROOT = OUTPUT_ROOT / "features"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
LOG_ROOT = OUTPUT_ROOT / "logs"

# ---------------------------------------------------------------------------
# ERA5 ARCO
# ---------------------------------------------------------------------------

ERA5_ARCO_URL = (
    "https://arco.datastores.ecmwf.int/"
    "cadl-arco-geo-002/arco/"
    "reanalysis_era5_single_levels/sfc/"
    "geoChunked.zarr"
)

# ---------------------------------------------------------------------------
# Historical range
# ---------------------------------------------------------------------------

START_YEAR = 1940

# "AUTO" means use the latest timestamp exposed by ERA5.
END_YEAR = "AUTO"

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

SAVE_RAW = True
SAVE_FEATURES = True

# Process one year at a time.
#
# This is deliberately conservative:
#
#     45 cities × 1 year
#
# instead of attempting the entire 1940-present cube.
#
CHUNK_YEARS = 1

# ---------------------------------------------------------------------------
# Existing files
# ---------------------------------------------------------------------------

OVERWRITE = False

# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------

MAX_RETRIES = 5
RETRY_BASE_SECONDS = 5

# ---------------------------------------------------------------------------
# ARCO access
# ---------------------------------------------------------------------------

# Your CDS key is read automatically from ~/.cdsapirc.
# We intentionally do not hard-code it.

CDS_RC = Path.home() / ".cdsapirc"


# ============================================================================
# SOALM CITY NETWORK
# ============================================================================

CITIES = {
    "Delhi": (28.6139, 77.2090, 216),
    "Mumbai": (19.0760, 72.8777, 14),
    "Bengaluru": (12.9716, 77.5946, 920),
    "Chennai": (13.0827, 80.2707, 6),
    "Hyderabad": (17.3850, 78.4867, 542),
    "Kolkata": (22.5726, 88.3639, 9),
    "Ahmedabad": (23.0225, 72.5714, 53),
    "Pune": (18.5204, 73.8567, 560),
    "Jaipur": (26.9124, 75.7873, 431),
    "Lucknow": (26.8467, 80.9462, 123),
    "Kanpur": (26.4499, 80.3319, 126),
    "Nagpur": (21.1458, 79.0882, 310),
    "Indore": (22.7196, 75.8577, 553),
    "Bhopal": (23.2599, 77.4126, 527),
    "Patna": (25.5941, 85.1376, 53),
    "Vadodara": (22.3072, 73.1812, 39),
    "Surat": (21.1702, 72.8311, 13),
    "Rajkot": (22.3039, 70.8022, 134),
    "Nashik": (19.9975, 73.7898, 565),
    "Agra": (27.1767, 78.0081, 170),
    "Varanasi": (25.3176, 82.9739, 81),
    "Prayagraj": (25.4358, 81.8463, 98),
    "Chandigarh": (30.7333, 76.7794, 321),
    "Amritsar": (31.6340, 74.8723, 234),
    "Dehradun": (30.3165, 78.0322, 640),
    "Shimla": (31.1048, 77.1734, 2276),
    "Srinagar": (34.0837, 74.7973, 1585),
    "Jammu": (32.7266, 74.8570, 327),
    "Raipur": (21.2514, 81.6296, 298),
    "Ranchi": (23.3441, 85.3096, 651),
    "Bhubaneswar": (20.2961, 85.8245, 45),
    "Cuttack": (20.4625, 85.8830, 24),
    "Guwahati": (26.1445, 91.7362, 55),
    "Shillong": (25.5788, 91.8933, 1496),
    "Siliguri": (26.7271, 88.3953, 122),
    "Kochi": (9.9312, 76.2673, 8),
    "Thiruvananthapuram": (8.5241, 76.9366, 10),
    "Coimbatore": (11.0168, 76.9558, 411),
    "Madurai": (9.9252, 78.1198, 101),
    "Mangaluru": (12.9141, 74.8560, 22),
    "Mysuru": (12.2958, 76.6394, 770),
    "Vijayawada": (16.5062, 80.6480, 11),
    "Visakhapatnam": (17.6868, 83.2185, 45),
    "Dibrugarh": (27.4728, 94.9120, 108),
    "Agartala": (23.8315, 91.2868, 12),
    "Imphal": (24.8170, 93.9368, 786),
}


# ============================================================================
# EXACT SOALM 40-FEATURE SCHEMA
# ============================================================================

SOALM_FEATURES = [
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
    "rain_last_1h",
    "rain_last_3h",
    "rain_last_6h",
    "rain_last_12h",
    "rain_last_24h",
    "rain_last_48h",
    "rain_last_72h",
    "pressure_change_1h",
    "pressure_change_3h",
    "pressure_change_6h",
    "temperature_change_3h",
    "temperature_change_6h",
    "humidity_change_3h",
    "humidity_change_6h",
    "wind_change_3h",
    "wind_change_6h",
    "latitude",
    "longitude",
    "elevation",
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


# ============================================================================
# DIRECTORY SETUP
# ============================================================================


def setup_directories():

    for directory in [
        OUTPUT_ROOT,
        RAW_ROOT,
        FEATURE_ROOT,
        CHECKPOINT_ROOT,
        LOG_ROOT,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CDS KEY
# ============================================================================


def load_cds_key():

    if not CDS_RC.exists():

        raise RuntimeError(
            "\n"
            "CDS configuration not found.\n\n"
            f"Expected:\n{CDS_RC}\n\n"
            "Create .cdsapirc with:\n\n"
            "url: https://cds.climate.copernicus.eu/api\n"
            "key: YOUR_API_KEY\n"
        )

    text = CDS_RC.read_text(encoding="utf-8", errors="ignore")

    key = None

    for line in text.splitlines():

        line = line.strip()

        if line.startswith("key:"):

            key = line.split(":", 1)[1].strip()

        elif line.startswith("key="):

            key = line.split("=", 1)[1].strip()

    if not key:

        raise RuntimeError(f"Could not find 'key:' inside {CDS_RC}")

    return key


# ============================================================================
# OPEN ARCO
# ============================================================================


def open_arco():

    print()
    print("=" * 78)
    print("OPENING ERA5 ARCO")
    print("=" * 78)

    cds_key = load_cds_key()

    print("✓ CDS API key found")
    print("✓ Using ECMWF ERA5 ARCO geo-chunked store")
    print()
    print("Opening metadata...")

    storage_options = {"headers": {"Authorization": f"Bearer {cds_key}"}}

    ds = xr.open_zarr(
        ERA5_ARCO_URL,
        consolidated=True,
        storage_options=storage_options,
    )

    print("✓ ARCO opened")
    print()
    print("Dataset:")
    print(ds)

    return ds


# ============================================================================
# VARIABLE RESOLUTION
# ============================================================================


def find_variable(ds, candidates):

    for name in candidates:

        if name in ds.variables:
            return name

        if name in ds.data_vars:
            return name

    return None


def resolve_variables(ds):

    mapping = {
        "temperature_2m": [
            "2m_temperature",
            "t2m",
        ],
        "dew_point_2m": [
            "2m_dewpoint_temperature",
            "d2m",
        ],
        "u10": [
            "10m_u_component_of_wind",
            "u10",
        ],
        "v10": [
            "10m_v_component_of_wind",
            "v10",
        ],
        "surface_pressure": [
            "surface_pressure",
            "sp",
        ],
        "precipitation": [
            "total_precipitation",
            "tp",
        ],
        "cloud_cover": [
            "total_cloud_cover",
            "tcc",
        ],
        "shortwave_radiation": [
            "surface_solar_radiation_downwards",
            "ssrd",
        ],
    }

    resolved = {}

    print()
    print("=" * 78)
    print("VARIABLE CHECK")
    print("=" * 78)

    for logical_name, candidates in mapping.items():

        actual = find_variable(ds, candidates)

        if actual is None:

            raise RuntimeError(
                f"\nRequired ERA5 variable unavailable: "
                f"{logical_name}\n"
                f"Tried: {candidates}\n"
            )

        resolved[logical_name] = actual

        print(f"✓ {logical_name:25s} → {actual}")

    return resolved


# ============================================================================
# YEAR RANGE
# ============================================================================


def get_available_time_range(ds):

    time_name = "time"

    if time_name not in ds.coords:

        raise RuntimeError(
            "ERA5 ARCO dataset does not expose expected 'time' coordinate."
        )

    first_time = pd.Timestamp(ds[time_name].values[0])

    last_time = pd.Timestamp(ds[time_name].values[-1])

    return first_time, last_time


# ============================================================================
# CITY GRID COORDINATE
# ============================================================================


def nearest_grid_point(ds, latitude, longitude):

    lat_name = "latitude"
    lon_name = "longitude"

    lat_values = np.asarray(ds[lat_name].values)

    lon_values = np.asarray(ds[lon_name].values)

    lat_index = int(np.abs(lat_values - latitude).argmin())

    lon_index = int(np.abs(lon_values - longitude).argmin())

    return (
        float(lat_values[lat_index]),
        float(lon_values[lon_index]),
    )


# ============================================================================
# RH CALCULATION
# ============================================================================


def calculate_relative_humidity(
    temperature_k,
    dewpoint_k,
):

    temperature_c = temperature_k - 273.15
    dewpoint_c = dewpoint_k - 273.15

    # Magnus approximation
    a = 17.625
    b = 243.04

    numerator = (a * dewpoint_c) / (b + dewpoint_c)

    denominator = (a * temperature_c) / (b + temperature_c)

    rh = 100.0 * np.exp(numerator - denominator)

    return np.clip(rh, 0.0, 100.0)


# ============================================================================
# WEATHER EXTRACTION
# ============================================================================


def extract_city_year(
    ds,
    variables,
    city,
    latitude,
    longitude,
    elevation,
    year,
):

    start = f"{year}-01-01T00:00:00"

    end = f"{year}-12-31T23:00:00"

    print(f"    → Extracting {city} {year}")

    grid_lat, grid_lon = nearest_grid_point(
        ds,
        latitude,
        longitude,
    )

    print(f"      requested: " f"{latitude:.4f}, {longitude:.4f}")

    print(f"      ERA5 grid: " f"{grid_lat:.2f}, {grid_lon:.2f}")

    selected = (
        ds[
            [
                variables["temperature_2m"],
                variables["dew_point_2m"],
                variables["u10"],
                variables["v10"],
                variables["surface_pressure"],
                variables["precipitation"],
                variables["cloud_cover"],
                variables["shortwave_radiation"],
            ]
        ]
        .sel(
            latitude=grid_lat,
            longitude=grid_lon,
            method="nearest",
        )
        .sel(time=slice(start, end))
    )

    # Force actual download of this city/year only.
    dataframe = selected.to_dataframe().reset_index()

    if dataframe.empty:

        raise RuntimeError(f"No ERA5 data returned for {city} {year}")

    rename_map = {
        variables["temperature_2m"]: "temperature_2m_raw",
        variables["dew_point_2m"]: "dew_point_2m_raw",
        variables["u10"]: "u10",
        variables["v10"]: "v10",
        variables["surface_pressure"]: "surface_pressure_raw",
        variables["precipitation"]: "precipitation_raw",
        variables["cloud_cover"]: "cloud_cover_raw",
        variables["shortwave_radiation"]: "shortwave_radiation_raw",
    }

    dataframe = dataframe.rename(columns=rename_map)

    dataframe["city"] = city

    dataframe["latitude"] = latitude

    dataframe["longitude"] = longitude

    dataframe["elevation"] = elevation

    return dataframe


# ============================================================================
# RAW NORMALIZATION
# ============================================================================


def normalize_raw(df):

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["time"],
        utc=True,
    )

    # Kelvin → Celsius
    df["temperature_2m"] = df["temperature_2m_raw"] - 273.15

    df["dew_point_2m"] = df["dew_point_2m_raw"] - 273.15

    # Relative humidity
    df["relative_humidity_2m"] = calculate_relative_humidity(
        df["temperature_2m_raw"].to_numpy(),
        df["dew_point_2m_raw"].to_numpy(),
    )

    # Wind
    u = df["u10"].to_numpy()

    v = df["v10"].to_numpy()

    df["wind_speed_10m"] = np.sqrt(u**2 + v**2)

    # Meteorological wind direction
    direction = (
        np.degrees(
            np.arctan2(
                -u,
                -v,
            )
        )
        + 360
    ) % 360

    df["wind_direction_10m"] = direction

    # ERA5 total precipitation is metres.
    #
    # Convert to mm.
    #
    df["precipitation"] = df["precipitation_raw"] * 1000.0

    # Existing SOALM naming:
    #
    # rain = precipitation.
    #
    df["rain"] = df["precipitation"].clip(lower=0)

    # ERA5 single-level ARCO does not expose
    # Open-Meteo's "showers" variable.
    #
    # Keep schema compatibility.
    #
    # We define showers as precipitation occurring
    # during the hour. This is deliberately documented
    # rather than pretending it is a native ERA5 variable.
    #
    df["showers"] = np.where(
        df["rain"] > 0,
        df["rain"],
        0.0,
    )

    # Cloud cover 0-1 → percent
    df["cloud_cover"] = (df["cloud_cover_raw"] * 100.0).clip(0, 100)

    # Surface pressure Pa → hPa
    df["surface_pressure"] = df["surface_pressure_raw"] / 100.0

    # Surface solar radiation:
    #
    # ERA5 provides J/m² for the hourly accumulation.
    #
    # Convert to W/m² using 3600 seconds.
    #
    df["shortwave_radiation"] = df["shortwave_radiation_raw"] / 3600.0

    # Remove duplicate timestamps.
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"],
        keep="first",
    )

    return df


# ============================================================================
# EXACT 40 FEATURE ENGINEERING
# ============================================================================


def build_soalm_features(df):

    df = df.copy()

    df = df.sort_values(["city", "timestamp"])

    grouped = df.groupby(
        "city",
        group_keys=False,
    )

    # -----------------------------------------------------------------------
    # Rain windows
    # -----------------------------------------------------------------------

    for hours in [
        1,
        3,
        6,
        12,
        24,
        48,
        72,
    ]:

        if hours == 1:

            df[f"rain_last_{hours}h"] = df["rain"]

        else:

            df[f"rain_last_{hours}h"] = grouped["rain"].transform(
                lambda x: x.rolling(
                    hours,
                    min_periods=1,
                ).sum()
            )

    # -----------------------------------------------------------------------
    # Pressure changes
    # -----------------------------------------------------------------------

    for hours in [1, 3, 6]:

        df[f"pressure_change_{hours}h"] = grouped["surface_pressure"].transform(
            lambda x: x.diff(hours)
        )

    # -----------------------------------------------------------------------
    # Temperature changes
    # -----------------------------------------------------------------------

    for hours in [3, 6]:

        df[f"temperature_change_{hours}h"] = grouped["temperature_2m"].transform(
            lambda x: x.diff(hours)
        )

    # -----------------------------------------------------------------------
    # Humidity changes
    # -----------------------------------------------------------------------

    for hours in [3, 6]:

        df[f"humidity_change_{hours}h"] = grouped["relative_humidity_2m"].transform(
            lambda x: x.diff(hours)
        )

    # -----------------------------------------------------------------------
    # Wind changes
    # -----------------------------------------------------------------------

    for hours in [3, 6]:

        df[f"wind_change_{hours}h"] = grouped["wind_speed_10m"].transform(
            lambda x: x.diff(hours)
        )

    # -----------------------------------------------------------------------
    # Calendar
    # -----------------------------------------------------------------------

    ts = df["timestamp"]

    df["hour"] = ts.dt.hour

    df["day_of_week"] = ts.dt.dayofweek

    df["month"] = ts.dt.month

    df["day_of_year"] = ts.dt.dayofyear

    # -----------------------------------------------------------------------
    # Cyclical hour
    # -----------------------------------------------------------------------

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)

    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # -----------------------------------------------------------------------
    # Cyclical month
    # -----------------------------------------------------------------------

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)

    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # -----------------------------------------------------------------------
    # Cyclical day of year
    # -----------------------------------------------------------------------

    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)

    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # -----------------------------------------------------------------------
    # Make sure every required feature exists
    # -----------------------------------------------------------------------

    missing = [feature for feature in SOALM_FEATURES if feature not in df.columns]

    if missing:

        raise RuntimeError("Missing SOALM features:\n" + "\n".join(missing))

    return df


# ============================================================================
# ATOMIC PARQUET SAVE
# ============================================================================


def atomic_parquet_save(df, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    if temporary.exists():
        temporary.unlink()

    df.to_parquet(
        temporary,
        index=False,
        compression="zstd",
    )

    temporary.replace(path)


# ============================================================================
# CHECKPOINT
# ============================================================================


def checkpoint_path(city, year):

    safe_city = city.lower().replace(
        " ",
        "_",
    )

    return CHECKPOINT_ROOT / f"{safe_city}_{year}.json"


def is_completed(city, year):

    return checkpoint_path(
        city,
        year,
    ).exists()


def mark_completed(city, year, rows):

    path = checkpoint_path(
        city,
        year,
    )

    payload = {
        "city": city,
        "year": year,
        "rows": int(rows),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================================
# SAVE RAW
# ============================================================================


def save_raw_city_year(
    df,
    city,
    year,
):

    city_dir = RAW_ROOT / city.lower().replace(" ", "_")

    city_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = city_dir / f"{year}.parquet"

    if path.exists() and not OVERWRITE:

        print(f"      ✓ raw exists: {path.name}")

        return

    raw_columns = [
        "timestamp",
        "city",
        "latitude",
        "longitude",
        "elevation",
        "temperature_2m",
        "dew_point_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "showers",
        "cloud_cover",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "shortwave_radiation",
    ]

    raw = df[[c for c in raw_columns if c in df.columns]].copy()

    atomic_parquet_save(
        raw,
        path,
    )


# ============================================================================
# SAVE FEATURES
# ============================================================================


def save_feature_city_year(
    df,
    city,
    year,
):

    city_dir = FEATURE_ROOT / city.lower().replace(" ", "_")

    city_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = city_dir / f"{year}.parquet"

    if path.exists() and not OVERWRITE:

        print(f"      ✓ features exist: {path.name}")

        return

    feature_df = df[
        [
            "timestamp",
            "city",
        ]
        + SOALM_FEATURES
    ].copy()

    atomic_parquet_save(
        feature_df,
        path,
    )


# ============================================================================
# PROCESS CITY/YEAR
# ============================================================================


def process_city_year(
    ds,
    variables,
    city,
    latitude,
    longitude,
    elevation,
    year,
):

    if is_completed(city, year):

        print(f"[SKIP] {city} {year} " f"already completed")

        return 0

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            raw = extract_city_year(
                ds,
                variables,
                city,
                latitude,
                longitude,
                elevation,
                year,
            )

            raw = normalize_raw(raw)

            features = build_soalm_features(raw)

            if SAVE_RAW:

                save_raw_city_year(
                    features,
                    city,
                    year,
                )

            if SAVE_FEATURES:

                save_feature_city_year(
                    features,
                    city,
                    year,
                )

            mark_completed(
                city,
                year,
                len(features),
            )

            print(f"    ✓ SUCCESS " f"{city} {year}: " f"{len(features):,} rows")

            return len(features)

        except KeyboardInterrupt:

            raise

        except Exception as exc:

            print(f"    ✗ attempt " f"{attempt}/{MAX_RETRIES}: " f"{exc}")

            if attempt >= MAX_RETRIES:

                print(f"    ✗ FAILED " f"{city} {year}")

                return 0

            wait = RETRY_BASE_SECONDS * (2 ** (attempt - 1))

            print(f"    → retrying in " f"{wait}s")

            time.sleep(wait)

    return 0


# ============================================================================
# VALIDATE FEATURE SCHEMA
# ============================================================================


def validate_schema():

    print()
    print("=" * 78)
    print("SOALM FEATURE SCHEMA")
    print("=" * 78)

    for index, feature in enumerate(
        SOALM_FEATURES,
        1,
    ):

        print(f"{index:02d}. {feature}")

    print()
    print(f"Total features: " f"{len(SOALM_FEATURES)}")

    if len(SOALM_FEATURES) != 40:

        raise RuntimeError("SOALM feature schema is not exactly 40 features.")


# ============================================================================
# SUMMARY
# ============================================================================


def write_summary(
    total_rows,
    successful_jobs,
    skipped_jobs,
    failed_jobs,
    start_time,
):

    runtime = (time.time() - start_time) / 3600.0

    summary = {
        "version": "V1.6",
        "cities": len(CITIES),
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "successful_jobs": successful_jobs,
        "skipped_jobs": skipped_jobs,
        "failed_jobs": failed_jobs,
        "rows": total_rows,
        "features": len(SOALM_FEATURES),
        "runtime_hours": runtime,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    path = OUTPUT_ROOT / "download_summary.json"

    path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("SOALM V1.6 — DOWNLOAD SUMMARY")
    print("=" * 78)

    print(f"Cities:             {len(CITIES)}")

    print(f"Successful jobs:    {successful_jobs:,}")

    print(f"Skipped jobs:       {skipped_jobs:,}")

    print(f"Failed jobs:        {failed_jobs:,}")

    print(f"Rows added:         {total_rows:,}")

    print(f"Runtime:            {runtime:.2f} hours")

    print()
    print(f"Features:            " f"{len(SOALM_FEATURES)}")

    print()
    print(f"Output:\n{OUTPUT_ROOT}")


# ============================================================================
# MAIN
# ============================================================================


def main():

    setup_directories()

    validate_schema()

    print()
    print("=" * 78)
    print("SOALM V1.6 — ERA5 ARCO HISTORICAL ACQUISITION")
    print("=" * 78)

    print()
    print(f"Cities:       {len(CITIES)}")

    print(f"Period:       {START_YEAR} → PRESENT")

    print(f"Features:     {len(SOALM_FEATURES)}")

    print()
    print("OUTPUT")
    print(f"  {OUTPUT_ROOT}")

    print()
    print("SAFETY")
    print("  ✓ data/city_downloads/ untouched")
    print("  ✓ data/extreme_downloads/ untouched")
    print("  ✓ ERA5 stored separately")
    print("  ✓ City/year checkpoints")
    print("  ✓ Atomic Parquet writes")
    print("  ✓ Resume supported")
    print("  ✓ Exact 40-feature SOALM schema")

    start_time = time.time()

    # -----------------------------------------------------------------------
    # Open ARCO
    # -----------------------------------------------------------------------

    try:

        ds = open_arco()

    except Exception as exc:

        print()
        print("=" * 78)
        print("ARCO INITIALIZATION FAILED")
        print("=" * 78)

        print()
        print(exc)

        print()
        print("This is usually an authentication/" "CDS-key issue.")

        print()
        print("Verify:")

        print(f"  {CDS_RC}")

        raise

    # -----------------------------------------------------------------------
    # Variables
    # -----------------------------------------------------------------------

    variables = resolve_variables(ds)

    # -----------------------------------------------------------------------
    # Time range
    # -----------------------------------------------------------------------

    first_time, last_time = get_available_time_range(ds)

    actual_start_year = max(
        START_YEAR,
        first_time.year,
    )

    actual_end_year = last_time.year

    print()
    print("=" * 78)
    print("ERA5 AVAILABLE RANGE")
    print("=" * 78)

    print(f"First: {first_time}")

    print(f"Last:  {last_time}")

    print(f"Using: {actual_start_year} → " f"{actual_end_year}")

    years = range(
        actual_start_year,
        actual_end_year + 1,
    )

    total_jobs = len(CITIES) * len(list(years))

    print()
    print(f"Potential city/year jobs: " f"{total_jobs:,}")

    print()
    print("NOTE:")

    print("The script does NOT download the " "global ERA5 cube.")

    print("It extracts only the 45 SOALM city " "locations.")

    print()

    successful_jobs = 0
    skipped_jobs = 0
    failed_jobs = 0
    total_rows = 0

    # -----------------------------------------------------------------------
    # CITY LOOP
    # -----------------------------------------------------------------------

    job_number = 0

    try:

        for city_index, (
            city,
            coordinates,
        ) in enumerate(
            CITIES.items(),
            1,
        ):

            latitude, longitude, elevation = coordinates

            print()
            print("=" * 78)
            print(f"CITY " f"{city_index}/{len(CITIES)} — " f"{city}")
            print("=" * 78)

            for year in years:

                job_number += 1

                if is_completed(
                    city,
                    year,
                ):

                    skipped_jobs += 1

                    print(f"[{job_number:,}/" f"{total_jobs:,}] " f"SKIP {city} {year}")

                    continue

                print()
                print(f"[{job_number:,}/" f"{total_jobs:,}] " f"{city} {year}")

                rows = process_city_year(
                    ds,
                    variables,
                    city,
                    latitude,
                    longitude,
                    elevation,
                    year,
                )

                if rows > 0:

                    successful_jobs += 1

                    total_rows += rows

                else:

                    failed_jobs += 1

    except KeyboardInterrupt:

        print()
        print("=" * 78)
        print("CTRL+C RECEIVED")
        print("=" * 78)

        print()
        print("All successfully completed " "city/year files remain safe.")

        print("Run the script again to resume.")

        write_summary(
            total_rows,
            successful_jobs,
            skipped_jobs,
            failed_jobs,
            start_time,
        )

        return

    # -----------------------------------------------------------------------
    # FINAL
    # -----------------------------------------------------------------------

    write_summary(
        total_rows,
        successful_jobs,
        skipped_jobs,
        failed_jobs,
        start_time,
    )

    print()
    print("=" * 78)
    print("SOALM V1.6 COMPLETE")
    print("=" * 78)

    print()
    print("Next step:")

    print("Merge data/era5/features/*/*.parquet " "into the SOALM training dataset.")


if __name__ == "__main__":

    main()
