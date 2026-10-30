"""
===============================================================================
SOALM V2 — OPTIMIZED 3-HOUR DATA PREPARATION
===============================================================================

RAW ERA5
   ↓
Clean + normalize
   ↓
Sort chronologically per city
   ↓
Calculate historical features
   ↓
Calculate future regression targets
   ↓
Calculate classification targets
   ↓
Keep every 3rd hour
   ↓
Preserve important/extreme weather events
   ↓
Optimized Parquet dataset

IMPORTANT:
- RAW data is NEVER modified.
- Future targets are calculated BEFORE 3-hour sampling.
- Processing is city-by-city to reduce RAM usage.
===============================================================================
"""

from pathlib import Path
import math
import gc
import sys
import traceback

import numpy as np
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "era5" / "raw"

OUTPUT_DIR = BASE_DIR / "data" / "era5" / "prepared"

OUTPUT_FILE = OUTPUT_DIR / "soalm_v2_3h.parquet"

FEATURE_FILE = BASE_DIR / "data" / "era5" / "features" / "soalm_v2_features.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_HOURS = 3

# Rain classification thresholds.
# Same basic thresholds used in the earlier SOALM pipeline.
LOW_THRESHOLD = 10.0
MODERATE_THRESHOLD = 30.0
HIGH_THRESHOLD = 60.0

# Event thresholds.
#
# These can be changed later without touching RAW data.
HEAVY_RAIN_THRESHOLD = 30.0
EXTREME_RAIN_THRESHOLD = 60.0

DANGEROUS_ACCUMULATION_THRESHOLD = 50.0
CRITICAL_ACCUMULATION_THRESHOLD = 100.0

# Intensification threshold in mm over the corresponding period.
INTENSIFICATION_6H_THRESHOLD = 20.0
INTENSIFICATION_24H_THRESHOLD = 50.0

# Compression
PARQUET_COMPRESSION = "zstd"


# =============================================================================
# SOALM FEATURES
# =============================================================================

FEATURES = [
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


# =============================================================================
# TARGETS
# =============================================================================

REGRESSION_TARGETS = [
    "rain_future_1h",
    "rain_future_3h",
    "rain_future_6h",
    "rain_future_12h",
    "rain_future_24h",
    "rain_future_48h",
    "rain_future_72h",
    "rain_accum_future_3h",
    "rain_accum_future_6h",
    "rain_accum_future_12h",
    "rain_accum_future_24h",
    "rain_accum_future_48h",
    "rain_accum_future_72h",
    "rain_max_future_3h",
    "rain_max_future_6h",
    "rain_max_future_12h",
    "rain_max_future_24h",
    "rain_max_future_48h",
    "rain_max_future_72h",
]


CLASSIFICATION_TARGETS = [
    "risk_1h",
    "risk_3h",
    "risk_6h",
    "risk_12h",
    "risk_24h",
    "heavy_rain_event_24h",
    "extreme_rain_event_24h",
    "dangerous_accumulation_24h",
    "critical_accumulation_24h",
    "rain_intensification_6h",
    "rain_intensification_24h",
]


ALL_TARGETS = REGRESSION_TARGETS + CLASSIFICATION_TARGETS


# =============================================================================
# EXPECTED RAW WEATHER VARIABLES
# =============================================================================

WEATHER_COLUMNS = [
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


# =============================================================================
# PRINT HEADER
# =============================================================================

print("=" * 80)
print("SOALM V2 — OPTIMIZED 3-HOUR DATA PREPARATION")
print("=" * 80)

print()
print("Project root:")
print(BASE_DIR)

print()
print("RAW directory:")
print(RAW_DIR)

print()
print("Output:")
print(OUTPUT_FILE)

print()
print("Sampling:")
print("Every 3 hours AFTER target generation")

print()
print("Features:")
print(len(FEATURES))

print("Regression targets:")
print(len(REGRESSION_TARGETS))

print("Classification targets:")
print(len(CLASSIFICATION_TARGETS))

print("Total targets:")
print(len(ALL_TARGETS))

print()
print("=" * 80)


# =============================================================================
# HELPERS
# =============================================================================


def clean_name(value):
    """
    Convert filenames/city names into a predictable string.
    """
    value = str(value).strip()

    if not value:
        return "UNKNOWN"

    return value


def find_timestamp_column(df):
    """
    Detect timestamp column.
    """

    candidates = [
        "timestamp",
        "time",
        "datetime",
        "date",
        "valid_time",
    ]

    lower_map = {str(column).lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    return None


def find_city_column(df):
    """
    Detect city/location column.
    """

    candidates = [
        "city",
        "location",
        "place",
        "station",
    ]

    lower_map = {str(column).lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    return None


def safe_numeric(df, columns):
    """
    Convert selected columns to numeric.
    """

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def get_column(df, name, default=0.0):
    """
    Return numeric column or create a zero-filled Series.
    """

    if name in df.columns:

        return pd.to_numeric(df[name], errors="coerce").fillna(default)

    return pd.Series(default, index=df.index, dtype="float32")


def classify_risk(value):
    """
    Convert future rainfall into SOALM risk category.
    """

    if pd.isna(value):
        return None

    value = float(value)

    if value < LOW_THRESHOLD:
        return "LOW"

    if value < MODERATE_THRESHOLD:
        return "MODERATE"

    if value < HIGH_THRESHOLD:
        return "HIGH"

    return "CRITICAL"


# =============================================================================
# LOAD RAW FILE
# =============================================================================


def read_raw_file(file_path):
    """
    Read CSV or Parquet.
    """

    suffix = file_path.suffix.lower()

    if suffix == ".csv":

        df = pd.read_csv(file_path, low_memory=False)

    elif suffix in [".parquet", ".pq"]:

        df = pd.read_parquet(file_path)

    else:

        raise ValueError(f"Unsupported file format: {file_path.name}")

    return df


# =============================================================================
# NORMALIZE RAW DATA
# =============================================================================


def normalize_raw(df, source_file):
    """
    Normalize timestamp, city, coordinates and weather variables.
    """

    timestamp_column = find_timestamp_column(df)

    if timestamp_column is None:

        raise ValueError(
            f"No timestamp/time column found in {source_file.name}. "
            f"Available columns: {list(df.columns)}"
        )

    # Rename timestamp consistently.
    if timestamp_column != "timestamp":

        df = df.rename(columns={timestamp_column: "timestamp"})

    # -------------------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------------------

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    df = df.dropna(subset=["timestamp"])

    # -------------------------------------------------------------------------
    # City
    # -------------------------------------------------------------------------

    city_column = find_city_column(df)

    if city_column is not None:

        if city_column != "city":

            df = df.rename(columns={city_column: "city"})

        df["city"] = df["city"].astype(str)

    else:

        # Use filename as fallback.
        df["city"] = source_file.stem

    # -------------------------------------------------------------------------
    # Weather variables
    # -------------------------------------------------------------------------

    df = safe_numeric(df, WEATHER_COLUMNS)

    # -------------------------------------------------------------------------
    # Latitude / longitude
    # -------------------------------------------------------------------------

    if "latitude" not in df.columns:

        df["latitude"] = np.nan

    if "longitude" not in df.columns:

        df["longitude"] = np.nan

    if "elevation" not in df.columns:

        df["elevation"] = 0.0

    df = safe_numeric(
        df,
        [
            "latitude",
            "longitude",
            "elevation",
        ],
    )

    # -------------------------------------------------------------------------
    # Rain
    # -------------------------------------------------------------------------

    if "rain" not in df.columns:

        if "precipitation" in df.columns:

            df["rain"] = df["precipitation"]

        else:

            df["rain"] = 0.0

    df["rain"] = pd.to_numeric(df["rain"], errors="coerce").fillna(0).clip(lower=0)

    # -------------------------------------------------------------------------
    # Sort
    # -------------------------------------------------------------------------

    df = df.sort_values(["city", "timestamp"])

    # Remove duplicate city/timestamp observations.

    df = df.drop_duplicates(subset=["city", "timestamp"], keep="last")

    return df.reset_index(drop=True)


# =============================================================================
# HISTORICAL FEATURES
# =============================================================================


def create_historical_features(df):
    """
    Calculate all 40 SOALM features.

    Calculations are performed separately for each city.
    """

    # -------------------------------------------------------------------------
    # Ensure base weather columns exist
    # -------------------------------------------------------------------------

    for column in WEATHER_COLUMNS:

        if column not in df.columns:

            df[column] = 0.0

    # -------------------------------------------------------------------------
    # Rain
    # -------------------------------------------------------------------------

    rain = get_column(df, "rain", 0.0)

    # Historical rainfall.

    for hours in [
        1,
        3,
        6,
        12,
        24,
        48,
        72,
    ]:

        df[f"rain_last_{hours}h"] = (
            rain.groupby(df["city"])
            .rolling(window=hours, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )

    # -------------------------------------------------------------------------
    # Changes
    # -------------------------------------------------------------------------

    def create_change(column, hours, output):

        series = get_column(df, column, 0.0)

        df[output] = series - series.groupby(df["city"]).shift(hours)

        df[output] = df[output].fillna(0).astype("float32")

    create_change("surface_pressure", 1, "pressure_change_1h")

    create_change("surface_pressure", 3, "pressure_change_3h")

    create_change("surface_pressure", 6, "pressure_change_6h")

    create_change("temperature_2m", 3, "temperature_change_3h")

    create_change("temperature_2m", 6, "temperature_change_6h")

    create_change("relative_humidity_2m", 3, "humidity_change_3h")

    create_change("relative_humidity_2m", 6, "humidity_change_6h")

    create_change("wind_speed_10m", 3, "wind_change_3h")

    create_change("wind_speed_10m", 6, "wind_change_6h")

    # -------------------------------------------------------------------------
    # Time features
    # -------------------------------------------------------------------------

    timestamp = df["timestamp"]

    df["hour"] = timestamp.dt.hour.astype("int8")

    df["day_of_week"] = timestamp.dt.dayofweek.astype("int8")

    df["month"] = timestamp.dt.month.astype("int8")

    df["day_of_year"] = timestamp.dt.dayofyear.astype("int16")

    # -------------------------------------------------------------------------
    # Cyclic features
    # -------------------------------------------------------------------------

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")

    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype("float32")

    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype("float32")

    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365).astype(
        "float32"
    )

    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365).astype(
        "float32"
    )

    return df


# =============================================================================
# FUTURE TARGET GENERATION
# =============================================================================


def create_future_targets(df):
    """
    Create all future rainfall targets BEFORE sampling.

    This is extremely important.

    We calculate from hourly ERA5 first and only afterwards
    select every third hour.
    """

    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    rain = get_column(df, "rain", 0.0)

    grouped = rain.groupby(df["city"])

    # =========================================================================
    # FUTURE SINGLE-HOUR RAIN
    # =========================================================================

    for hours in [
        1,
        3,
        6,
        12,
        24,
        48,
        72,
    ]:

        df[f"rain_future_{hours}h"] = grouped.shift(-hours).astype("float32")

    # =========================================================================
    # FUTURE ACCUMULATION
    # =========================================================================
    #
    # For time t:
    #
    # future 3h = t+1 + t+2 + t+3
    #
    # =========================================================================

    for hours in [
        3,
        6,
        12,
        24,
        48,
        72,
    ]:

        shifted_values = []

        for offset in range(1, hours + 1):

            shifted_values.append(grouped.shift(-offset))

        accumulation = sum(shifted_values)

        df[f"rain_accum_future_{hours}h"] = accumulation.astype("float32")

    # =========================================================================
    # FUTURE MAXIMUM RAIN
    # =========================================================================

    for hours in [
        3,
        6,
        12,
        24,
        48,
        72,
    ]:

        shifted_values = []

        for offset in range(1, hours + 1):

            shifted_values.append(grouped.shift(-offset))

        future_max = pd.concat(shifted_values, axis=1).max(axis=1)

        df[f"rain_max_future_{hours}h"] = future_max.astype("float32")

    # =========================================================================
    # RISK CLASSIFICATION
    # =========================================================================

    df["risk_1h"] = df["rain_future_1h"].apply(classify_risk)

    df["risk_3h"] = df["rain_future_3h"].apply(classify_risk)

    df["risk_6h"] = df["rain_future_6h"].apply(classify_risk)

    df["risk_12h"] = df["rain_future_12h"].apply(classify_risk)

    df["risk_24h"] = df["rain_future_24h"].apply(classify_risk)

    # =========================================================================
    # EVENT TARGETS
    # =========================================================================

    df["heavy_rain_event_24h"] = (
        df["rain_accum_future_24h"] >= HEAVY_RAIN_THRESHOLD
    ).astype("int8")

    df["extreme_rain_event_24h"] = (
        df["rain_accum_future_24h"] >= EXTREME_RAIN_THRESHOLD
    ).astype("int8")

    df["dangerous_accumulation_24h"] = (
        df["rain_accum_future_24h"] >= DANGEROUS_ACCUMULATION_THRESHOLD
    ).astype("int8")

    df["critical_accumulation_24h"] = (
        df["rain_accum_future_24h"] >= CRITICAL_ACCUMULATION_THRESHOLD
    ).astype("int8")

    # =========================================================================
    # RAIN INTENSIFICATION
    # =========================================================================

    current_rain = rain

    previous_6h = current_rain.groupby(df["city"]).shift(6).fillna(0)

    previous_24h = current_rain.groupby(df["city"]).shift(24).fillna(0)

    # Current rainfall compared with rainfall 6h ago.

    df["rain_intensification_6h"] = (
        current_rain - previous_6h >= INTENSIFICATION_6H_THRESHOLD
    ).astype("int8")

    df["rain_intensification_24h"] = (
        current_rain - previous_24h >= INTENSIFICATION_24H_THRESHOLD
    ).astype("int8")

    return df


# =============================================================================
# IMPORTANT EVENT DETECTION
# =============================================================================


def important_event_mask(df):
    """
    Preserve important weather events even when sampling every 3 hours.

    This prevents rare extreme events from disappearing completely.
    """

    mask = pd.Series(False, index=df.index)

    # Current heavy rain

    rain = get_column(df, "rain", 0.0)

    mask |= rain >= HEAVY_RAIN_THRESHOLD

    # Future heavy/extreme rainfall

    mask |= df["rain_accum_future_24h"] >= HEAVY_RAIN_THRESHOLD

    mask |= df["rain_accum_future_24h"] >= EXTREME_RAIN_THRESHOLD

    # Dangerous/critical accumulation

    mask |= df["rain_accum_future_24h"] >= DANGEROUS_ACCUMULATION_THRESHOLD

    mask |= df["rain_accum_future_24h"] >= CRITICAL_ACCUMULATION_THRESHOLD

    # Strong intensification

    mask |= df["rain_intensification_6h"] == 1

    mask |= df["rain_intensification_24h"] == 1

    return mask


# =============================================================================
# 3-HOUR SAMPLING
# =============================================================================


def optimize_sampling(df):
    """
    Keep every 3rd hourly observation.

    Base:
        timestamp hour % 3 == 0

    Plus:
        important/extreme event observations.
    """

    normal_mask = df["timestamp"].dt.hour % SAMPLE_HOURS == 0

    event_mask = important_event_mask(df)

    final_mask = normal_mask | event_mask

    optimized = df.loc[final_mask].copy()

    return optimized


# =============================================================================
# FINAL COLUMN ORDER
# =============================================================================


def arrange_columns(df):

    metadata_columns = [
        "timestamp",
        "city",
    ]

    columns = metadata_columns + FEATURES + ALL_TARGETS

    # Remove duplicates while preserving order.

    columns = list(dict.fromkeys(columns))

    missing = [column for column in columns if column not in df.columns]

    if missing:

        raise RuntimeError("Missing final columns:\n" + "\n".join(missing))

    return df[columns]


# =============================================================================
# SAVE FEATURES
# =============================================================================


def save_feature_metadata():

    with open(FEATURE_FILE, "w", encoding="utf-8") as f:

        f.write("SOALM V2 FEATURES\n")

        f.write("=================\n\n")

        for index, feature in enumerate(FEATURES, start=1):

            f.write(f"{index:02d}. {feature}\n")

        f.write("\n\nREGRESSION TARGETS\n")

        f.write("==================\n\n")

        for index, target in enumerate(REGRESSION_TARGETS, start=1):

            f.write(f"{index:02d}. {target}\n")

        f.write("\n\nCLASSIFICATION TARGETS\n")

        f.write("======================\n\n")

        for index, target in enumerate(CLASSIFICATION_TARGETS, start=1):

            f.write(f"{index:02d}. {target}\n")

    print(f"✓ Feature metadata saved: {FEATURE_FILE}")


# =============================================================================
# DISCOVER RAW FILES
# =============================================================================


def discover_files():

    if not RAW_DIR.exists():

        raise FileNotFoundError(f"RAW directory does not exist:\n{RAW_DIR}")

    files = []

    for extension in [
        "*.csv",
        "*.parquet",
        "*.pq",
    ]:

        files.extend(RAW_DIR.rglob(extension))

    files = sorted(set(files))

    return files


# =============================================================================
# BUILD CITY DATASET
# =============================================================================


def load_all_raw_data():

    files = discover_files()

    if not files:

        raise RuntimeError(f"No CSV/Parquet files found in:\n{RAW_DIR}")

    print()
    print("=" * 80)
    print("RAW FILES")
    print("=" * 80)

    print(f"Files found: {len(files)}")

    for file in files:

        print(f"  - {file.relative_to(BASE_DIR)}")

    print()

    # -------------------------------------------------------------------------
    # We don't immediately concatenate everything.
    #
    # Instead we determine which cities occur in which files.
    # -------------------------------------------------------------------------

    city_frames = {}

    for file_index, file_path in enumerate(files, start=1):

        print(f"[{file_index}/{len(files)}] " f"Reading {file_path.name}")

        try:

            df = read_raw_file(file_path)

            print(f"    Raw rows: {len(df):,}")

            df = normalize_raw(df, file_path)

            cities = df["city"].dropna().astype(str).unique()

            print(f"    Cities: {len(cities)}")

            for city in cities:

                city_df = df[df["city"] == city].copy()

                city_frames.setdefault(city, []).append(city_df)

            del df

            gc.collect()

        except Exception as e:

            print()
            print(f"✗ Failed: {file_path}")

            print(repr(e))

            traceback.print_exc()

            raise

    return city_frames


# =============================================================================
# PROCESS CITY
# =============================================================================


def process_city(city, frames):

    print()
    print("=" * 80)
    print(f"PROCESSING CITY: {city}")
    print("=" * 80)

    df = pd.concat(frames, ignore_index=True)

    del frames

    # -------------------------------------------------------------------------
    # Sort
    # -------------------------------------------------------------------------

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove duplicates.

    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    print(f"Hourly rows: {len(df):,}")

    if len(df) == 0:

        return None

    # -------------------------------------------------------------------------
    # Validate time spacing
    # -------------------------------------------------------------------------

    time_diff = df["timestamp"].diff().dropna()

    if len(time_diff) > 0:

        median_hours = time_diff.dt.total_seconds().median() / 3600

        print(f"Median timestep: {median_hours:.2f} hours")

    # -------------------------------------------------------------------------
    # Features
    # -------------------------------------------------------------------------

    print("Creating historical features...")

    df = create_historical_features(df)

    print("✓ 40 features generated")

    # -------------------------------------------------------------------------
    # Targets
    # -------------------------------------------------------------------------

    print("Creating future targets...")

    df = create_future_targets(df)

    print("✓ 30 targets generated")

    # -------------------------------------------------------------------------
    # Remove rows without complete future horizon.
    #
    # The final 72 hours cannot have complete 72-hour targets.
    # -------------------------------------------------------------------------

    before = len(df)

    required_future = [
        "rain_future_72h",
        "rain_accum_future_72h",
        "rain_max_future_72h",
    ]

    df = df.dropna(subset=required_future)

    removed = before - len(df)

    print(f"Removed incomplete future rows: {removed:,}")

    # -------------------------------------------------------------------------
    # Optimized 3-hour sampling
    # -------------------------------------------------------------------------

    before_sampling = len(df)

    df = optimize_sampling(df)

    print()
    print(f"Before 3-hour optimization: " f"{before_sampling:,}")

    print(f"After 3-hour optimization:  " f"{len(df):,}")

    if before_sampling > 0:

        percentage = len(df) / before_sampling * 100

        print(f"Retained: {percentage:.2f}%")

    # -------------------------------------------------------------------------
    # Final columns
    # -------------------------------------------------------------------------

    df = arrange_columns(df)

    # -------------------------------------------------------------------------
    # Downcast numeric columns
    # -------------------------------------------------------------------------

    for column in df.columns:

        if column in [
            "timestamp",
            "city",
            "risk_1h",
            "risk_3h",
            "risk_6h",
            "risk_12h",
            "risk_24h",
        ]:

            continue

        if pd.api.types.is_float_dtype(df[column]):

            df[column] = df[column].astype("float32")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print()
    print(f"Final city rows: {len(df):,}")

    return df


# =============================================================================
# MAIN
# =============================================================================


def main():

    print()
    print("Starting SOALM V2 preparation...")

    # =========================================================================
    # SAFETY
    # =========================================================================

    if OUTPUT_FILE.exists():

        print()
        print("WARNING: output dataset already exists:")

        print(OUTPUT_FILE)

        answer = input("\nOverwrite it? [y/N]: ").strip().lower()

        if answer != "y":

            print("Cancelled.")

            return

        OUTPUT_FILE.unlink()

    # =========================================================================
    # METADATA
    # =========================================================================

    save_feature_metadata()

    # =========================================================================
    # LOAD RAW
    # =========================================================================

    city_frames = load_all_raw_data()

    print()
    print("=" * 80)
    print("CITY SUMMARY")
    print("=" * 80)

    print(f"Cities discovered: {len(city_frames)}")

    # =========================================================================
    # PROCESS CITIES
    # =========================================================================

    all_processed = []

    total_raw_rows = 0
    total_processed_rows = 0

    for index, (city, frames) in enumerate(city_frames.items(), start=1):

        print()
        print(f"GLOBAL CITY [{index}/{len(city_frames)}]")

        raw_city_rows = sum(len(frame) for frame in frames)

        total_raw_rows += raw_city_rows

        processed = process_city(city, frames)

        if processed is not None:

            total_processed_rows += len(processed)

            all_processed.append(processed)

        del processed

        gc.collect()

    del city_frames

    # =========================================================================
    # COMBINE
    # =========================================================================

    print()
    print("=" * 80)
    print("COMBINING PREPARED DATA")
    print("=" * 80)

    if not all_processed:

        raise RuntimeError("No processed data was generated.")

    final_df = pd.concat(all_processed, ignore_index=True)

    del all_processed

    gc.collect()

    # =========================================================================
    # FINAL SORT
    # =========================================================================

    final_df = final_df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    # =========================================================================
    # FINAL VALIDATION
    # =========================================================================

    print()
    print("=" * 80)
    print("FINAL VALIDATION")
    print("=" * 80)

    expected_columns = ["timestamp", "city"] + FEATURES + ALL_TARGETS

    missing = [column for column in expected_columns if column not in final_df.columns]

    if missing:

        raise RuntimeError("Missing final columns:\n" + "\n".join(missing))

    print("✓ All required columns present")

    print(f"Rows: {len(final_df):,}")

    print(f"Columns: {len(final_df.columns)}")

    print(f"Cities: {final_df['city'].nunique():,}")

    # =========================================================================
    # TARGET COVERAGE
    # =========================================================================

    print()
    print("Target coverage:")

    for target in ALL_TARGETS:

        valid = final_df[target].notna().sum()

        percentage = valid / len(final_df) * 100 if len(final_df) else 0

        print(f"  {target:<40} " f"{valid:>12,} " f"({percentage:>6.2f}%)")

    # =========================================================================
    # CLASS DISTRIBUTIONS
    # =========================================================================

    print()
    print("=" * 80)
    print("CLASSIFICATION DISTRIBUTIONS")
    print("=" * 80)

    for target in CLASSIFICATION_TARGETS:

        print()
        print(target)

        counts = final_df[target].value_counts(dropna=False).sort_index()

        for label, count in counts.items():

            percentage = count / len(final_df) * 100

            print(f"  {str(label):<12} " f"{count:>12,} " f"({percentage:>8.4f}%)")

    # =========================================================================
    # MEMORY
    # =========================================================================

    memory_gb = final_df.memory_usage(deep=True).sum() / (1024**3)

    print()
    print(f"Approximate DataFrame memory: " f"{memory_gb:.2f} GB")

    # =========================================================================
    # WRITE PARQUET
    # =========================================================================

    print()
    print("=" * 80)
    print("WRITING OPTIMIZED DATASET")
    print("=" * 80)

    print("Compression:", PARQUET_COMPRESSION)

    print("Output:", OUTPUT_FILE)

    final_df.to_parquet(
        OUTPUT_FILE,
        engine="pyarrow",
        compression=PARQUET_COMPRESSION,
        index=False,
    )

    # =========================================================================
    # FILE SIZE
    # =========================================================================

    file_size_gb = OUTPUT_FILE.stat().st_size / (1024**3)

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    print()
    print("=" * 80)
    print("SOALM V2 — DATA PREPARATION COMPLETE")
    print("=" * 80)

    print()
    print(f"Original hourly rows:     {total_raw_rows:,}")

    print(f"Optimized rows:            {total_processed_rows:,}")

    if total_raw_rows > 0:

        retention = total_processed_rows / total_raw_rows * 100

        print(f"Data retained:             {retention:.2f}%")

    print()
    print(f"Cities:                    " f"{final_df['city'].nunique():,}")

    print(f"Features:                  {len(FEATURES)}")

    print(f"Regression targets:        " f"{len(REGRESSION_TARGETS)}")

    print(f"Classification targets:    " f"{len(CLASSIFICATION_TARGETS)}")

    print(f"Total targets:             " f"{len(ALL_TARGETS)}")

    print()
    print(f"Prepared dataset size:     " f"{file_size_gb:.2f} GB")

    print()
    print("Dataset:")

    print(OUTPUT_FILE)

    print()
    print("✓ RAW data was not modified")

    print("✓ Features calculated before sampling")

    print("✓ Future targets calculated before sampling")

    print("✓ Every 3-hour observation retained")

    print("✓ Important weather events retained")

    print("✓ 40 SOALM features available")

    print("✓ 30 SOALM targets available")

    print()
    print("=" * 80)
    print("NEXT STEP:")

    print("Train the 19 regression + 11 classification models")

    print("using the prepared Parquet dataset.")

    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print("Process interrupted by user.")

        sys.exit(1)

    except Exception as e:

        print()
        print("=" * 80)
        print("FATAL ERROR")
        print("=" * 80)

        print(repr(e))

        traceback.print_exc()

        sys.exit(1)
