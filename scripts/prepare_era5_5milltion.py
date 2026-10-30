"""
===============================================================================
SOALM V2 — CREATE 5 MILLION ROW TRAINING DATASET
===============================================================================

INPUT:
    data/era5/prepared/soalm_v2_3h.parquet

OUTPUT:
    data/era5/prepared/soalm_v2_5m.parquet

PURPOSE:
    Create a maximum 5-million-row training dataset from the existing
    prepared dataset.

DESIGN:
    - Original prepared dataset is NEVER modified.
    - Important/extreme weather events are preserved.
    - Remaining normal observations are randomly sampled.
    - Final dataset is capped at exactly 5,000,000 rows when possible.
    - All 40 features are preserved.
    - All 30 targets are preserved.
    - Memory optimized for a 16 GB RAM machine.
===============================================================================
"""

from pathlib import Path
import gc
import sys
import traceback

import numpy as np
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "era5" / "prepared" / "soalm_v2_3h.parquet"

OUTPUT_FILE = BASE_DIR / "data" / "era5" / "prepared" / "soalm_v2_5m.parquet"


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_ROWS = 5_000_000

RANDOM_STATE = 42

COMPRESSION = "zstd"


# =============================================================================
# FEATURES
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
# EVENT THRESHOLDS
# =============================================================================

HEAVY_RAIN_THRESHOLD = 30.0

EXTREME_RAIN_THRESHOLD = 60.0

DANGEROUS_ACCUMULATION_THRESHOLD = 50.0

CRITICAL_ACCUMULATION_THRESHOLD = 100.0


# =============================================================================
# HEADER
# =============================================================================


def print_header():

    print()
    print("=" * 80)
    print("SOALM V2 — 5 MILLION ROW DATASET CREATOR")
    print("=" * 80)

    print()
    print("Input:")
    print(INPUT_FILE)

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print("Target rows:")
    print(f"{TARGET_ROWS:,}")

    print()
    print("Random state:")
    print(RANDOM_STATE)

    print()
    print("=" * 80)


# =============================================================================
# VALIDATE INPUT
# =============================================================================


def validate_input():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(f"\nPrepared dataset not found:\n{INPUT_FILE}")

    print()
    print("✓ Input dataset found")

    print(
        "Disk size:",
        round(INPUT_FILE.stat().st_size / (1024**3), 2),
        "GB",
    )


# =============================================================================
# LOAD DATA
# =============================================================================


def load_dataset():

    print()
    print("=" * 80)
    print("LOADING PREPARED DATASET")
    print("=" * 80)

    print()
    print("Reading Parquet...")

    df = pd.read_parquet(INPUT_FILE)

    print()
    print("✓ Dataset loaded")

    print()
    print("Rows:", f"{len(df):,}")

    print("Columns:", len(df.columns))

    # -------------------------------------------------------------------------
    # Check required columns
    # -------------------------------------------------------------------------

    required_columns = ["timestamp", "city"] + FEATURES + ALL_TARGETS

    missing = [column for column in required_columns if column not in df.columns]

    if missing:

        raise RuntimeError("Missing required columns:\n" + "\n".join(missing))

    print()
    print("✓ All 40 features available")

    print("✓ All 30 targets available")

    return df


# =============================================================================
# MEMORY OPTIMIZATION
# =============================================================================


def optimize_memory(df):

    print()
    print("Optimizing numeric memory...")

    before = df.memory_usage(deep=True).sum() / (1024**3)

    for column in df.columns:

        if column in ["timestamp", "city"]:

            continue

        if pd.api.types.is_float_dtype(df[column]):

            df[column] = df[column].astype("float32")

        elif pd.api.types.is_integer_dtype(df[column]):

            df[column] = pd.to_numeric(
                df[column],
                downcast="integer",
            )

    after = df.memory_usage(deep=True).sum() / (1024**3)

    print(f"Memory: {before:.2f} GB -> {after:.2f} GB")

    return df


# =============================================================================
# IMPORTANT EVENT MASK
# =============================================================================


def create_event_mask(df):

    print()
    print("=" * 80)
    print("IDENTIFYING IMPORTANT WEATHER EVENTS")
    print("=" * 80)

    mask = pd.Series(
        False,
        index=df.index,
    )

    # -------------------------------------------------------------------------
    # Current rainfall
    # -------------------------------------------------------------------------

    if "rain" in df.columns:

        rain = pd.to_numeric(
            df["rain"],
            errors="coerce",
        ).fillna(0)

        mask |= rain >= HEAVY_RAIN_THRESHOLD

    # -------------------------------------------------------------------------
    # Future accumulation
    # -------------------------------------------------------------------------

    if "rain_accum_future_24h" in df.columns:

        accumulation = pd.to_numeric(
            df["rain_accum_future_24h"],
            errors="coerce",
        ).fillna(0)

        mask |= accumulation >= HEAVY_RAIN_THRESHOLD

        mask |= accumulation >= EXTREME_RAIN_THRESHOLD

        mask |= accumulation >= DANGEROUS_ACCUMULATION_THRESHOLD

        mask |= accumulation >= CRITICAL_ACCUMULATION_THRESHOLD

    # -------------------------------------------------------------------------
    # Classification events
    # -------------------------------------------------------------------------

    event_columns = [
        "heavy_rain_event_24h",
        "extreme_rain_event_24h",
        "dangerous_accumulation_24h",
        "critical_accumulation_24h",
        "rain_intensification_6h",
        "rain_intensification_24h",
    ]

    for column in event_columns:

        if column in df.columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            ).fillna(0)

            mask |= values == 1

    event_count = int(mask.sum())

    print()
    print(
        "Important/event rows:",
        f"{event_count:,}",
    )

    print(
        "Normal rows:",
        f"{len(df) - event_count:,}",
    )

    if len(df) > 0:

        print(
            "Event percentage:",
            f"{event_count / len(df) * 100:.2f}%",
        )

    return mask


# =============================================================================
# SAMPLE DATA
# =============================================================================


def create_training_subset(df):

    print()
    print("=" * 80)
    print("CREATING 5 MILLION ROW TRAINING DATASET")
    print("=" * 80)

    total_rows = len(df)

    # =========================================================================
    # CASE 1 — Dataset already <= 5M
    # =========================================================================

    if total_rows <= TARGET_ROWS:

        print()
        print(
            "Dataset already contains",
            f"{total_rows:,}",
            "rows.",
        )

        print("No sampling required.")

        return df.copy()

    # =========================================================================
    # EVENT DETECTION
    # =========================================================================

    event_mask = create_event_mask(df)

    event_df = df.loc[event_mask].copy()

    normal_df = df.loc[~event_mask].copy()

    # =========================================================================
    # TOO MANY EVENTS
    # =========================================================================

    if len(event_df) >= TARGET_ROWS:

        print()
        print(
            "WARNING: Important events alone exceed",
            f"{TARGET_ROWS:,}",
            "rows.",
        )

        print("Sampling events while preserving their distribution.")

        result = event_df.sample(
            n=TARGET_ROWS,
            random_state=RANDOM_STATE,
        ).copy()

    else:

        # =====================================================================
        # KEEP ALL EVENTS
        # =====================================================================

        events_to_keep = len(event_df)

        normal_needed = TARGET_ROWS - events_to_keep

        print()
        print(
            "Keeping all important events:",
            f"{events_to_keep:,}",
        )

        print(
            "Normal rows required:",
            f"{normal_needed:,}",
        )

        # =====================================================================
        # SAMPLE NORMAL DATA
        # =====================================================================

        if len(normal_df) > normal_needed:

            print()
            print("Sampling normal observations...")

            normal_sample = normal_df.sample(
                n=normal_needed,
                random_state=RANDOM_STATE,
            )

        else:

            normal_sample = normal_df

        # =====================================================================
        # COMBINE
        # =====================================================================

        result = pd.concat(
            [
                event_df,
                normal_sample,
            ],
            ignore_index=True,
        )

    # =========================================================================
    # SHUFFLE
    # =========================================================================

    print()
    print("Shuffling final dataset...")

    result = result.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    # =========================================================================
    # FINAL CAP
    # =========================================================================

    if len(result) > TARGET_ROWS:

        result = result.iloc[:TARGET_ROWS].copy()

    print()
    print("=" * 80)
    print("SUBSET CREATED")
    print("=" * 80)

    print()
    print(
        "Original rows:",
        f"{total_rows:,}",
    )

    print(
        "Final rows:",
        f"{len(result):,}",
    )

    print(
        "Reduction:",
        f"{(1 - len(result) / total_rows) * 100:.2f}%",
    )

    return result


# =============================================================================
# FINAL COLUMN ORDER
# =============================================================================


def arrange_columns(df):

    columns = ["timestamp", "city"] + FEATURES + ALL_TARGETS

    return df[columns]


# =============================================================================
# SAVE DATASET
# =============================================================================


def save_dataset(df):

    print()
    print("=" * 80)
    print("SAVING 5 MILLION ROW DATASET")
    print("=" * 80)

    if OUTPUT_FILE.exists():

        print()
        print("Output already exists:")

        print(OUTPUT_FILE)

        answer = input("\nOverwrite it? [y/N]: ").strip().lower()

        if answer != "y":

            print()
            print("Cancelled.")

            return False

        OUTPUT_FILE.unlink()

    # -------------------------------------------------------------------------
    # Final column order
    # -------------------------------------------------------------------------

    df = arrange_columns(df)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    print()
    print("Writing Parquet...")

    df.to_parquet(
        OUTPUT_FILE,
        engine="pyarrow",
        compression=COMPRESSION,
        index=False,
    )

    # -------------------------------------------------------------------------
    # File size
    # -------------------------------------------------------------------------

    size_gb = OUTPUT_FILE.stat().st_size / (1024**3)

    print()
    print("✓ Dataset saved")

    print()
    print("File:")
    print(OUTPUT_FILE)

    print()
    print("Rows:")
    print(f"{len(df):,}")

    print()
    print("Columns:")
    print(len(df.columns))

    print()
    print("Features:")
    print(len(FEATURES))

    print()
    print("Targets:")
    print(len(ALL_TARGETS))

    print()
    print("File size:")
    print(f"{size_gb:.2f} GB")

    return True


# =============================================================================
# MAIN
# =============================================================================


def main():

    print_header()

    start_time = pd.Timestamp.now()

    try:

        # ---------------------------------------------------------------------
        # Validate
        # ---------------------------------------------------------------------

        validate_input()

        # ---------------------------------------------------------------------
        # Load
        # ---------------------------------------------------------------------

        df = load_dataset()

        # ---------------------------------------------------------------------
        # Optimize memory
        # ---------------------------------------------------------------------

        df = optimize_memory(df)

        # ---------------------------------------------------------------------
        # Create 5M subset
        # ---------------------------------------------------------------------

        training_df = create_training_subset(df)

        # ---------------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------------

        save_dataset(training_df)

        # ---------------------------------------------------------------------
        # Cleanup
        # ---------------------------------------------------------------------

        del training_df
        del df

        gc.collect()

        # ---------------------------------------------------------------------
        # Runtime
        # ---------------------------------------------------------------------

        elapsed = (pd.Timestamp.now() - start_time).total_seconds()

        print()
        print("=" * 80)
        print("COMPLETE")
        print("=" * 80)

        print()
        print(
            "Runtime:",
            round(elapsed / 60, 2),
            "minutes",
        )

        print()
        print("Your training dataset is now:")

        print(OUTPUT_FILE)

        print()
        print(
            "Maximum rows:",
            f"{TARGET_ROWS:,}",
        )

        print()
        print("Next step:")

        print("Train the 19 regression + 11 classification models")

        print("using soalm_v2_5m.parquet.")

        print()
        print("=" * 80)

    except KeyboardInterrupt:

        print()
        print("=" * 80)
        print("PROCESS INTERRUPTED")
        print("=" * 80)

        sys.exit(1)

    except Exception as e:

        print()
        print("=" * 80)
        print("FATAL ERROR")
        print("=" * 80)

        print()
        print(repr(e))

        traceback.print_exc()

        sys.exit(1)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
