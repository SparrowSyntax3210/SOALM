"""
===============================================================================
SOALM — FINAL SURAT MULTI-SOURCE DATASET
===============================================================================

SOURCES
-------
1. ERA5
2. Global Flood Database (GDF)
3. Sentinel-1

PURPOSE
-------
Create one final Surat training dataset by combining:

    ERA5 weather/environmental features
        +
    GDF observed flood events
        +
    Sentinel-1 satellite observation metadata

IMPORTANT
---------
Sentinel-1 began in 2014.

Therefore Sentinel-1 columns will be empty/0 for older GDF/ERA5 periods.

This is EXPECTED.

The current Sentinel-1 index contains product metadata, NOT actual SAR
backscatter measurements.

Actual VV/VH/water-extent features can be added later.

OUTPUT
------
data/era5/prepared/surat_final_multisource.parquet


DATASET LOGIC
-------------
ERA5
    ↓
Hourly environmental observations

GDF
    ↓
Flood event intervals
    ↓
Hourly flood labels

Sentinel-1
    ↓
Satellite acquisitions
    ↓
Nearest observation / event-window matching

FINAL
    ↓
One row per ERA5 timestamp
    ↓
ERA5 + GDF + Sentinel-1
    ↓
Flood severity
    ↓
SOALM training dataset

===============================================================================
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import traceback

import numpy as np
import pandas as pd

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ERA5_ROOT = PROJECT_ROOT / "data" / "era5" / "features" / "surat"

GDF_FILE = (
    PROJECT_ROOT / "data" / "flood_training" / "surat" / "raw" / "gfd" / "surat.csv"
)

SENTINEL_INDEX = (
    PROJECT_ROOT / "data" / "sentinel1" / "surat" / "sentinel1_index.parquet"
)

OUTPUT_ROOT = PROJECT_ROOT / "data" / "era5" / "prepared"

OUTPUT_FILE = OUTPUT_ROOT / "surat_final_multisource.parquet"


# =============================================================================
# CONFIGURATION
# =============================================================================

# -------------------------------------------------------------------------
# Sentinel matching
# -------------------------------------------------------------------------

# A Sentinel acquisition can be associated with an ERA5 timestamp when it
# occurs within this many hours.
#
# Sentinel-1 is not hourly, so we DO NOT pretend that every ERA5 row has
# satellite data.
#
# Instead:
#
#     sentinel_available = 1
#
# means a real Sentinel-1 acquisition exists close to that timestamp.

MAX_SENTINEL_DISTANCE_HOURS = 72


# -------------------------------------------------------------------------
# Flood severity thresholds
# -------------------------------------------------------------------------
#
# These are intentionally conservative and transparent.
#
# flood_fraction:
#
# 0       = no mapped flood
# < 0.05  = low
# < 0.20  = moderate
# < 0.50  = high
# >= 0.50 = critical
#
# These can later be calibrated against the actual Surat GDF distribution.

LOW_THRESHOLD = 0.0
MODERATE_THRESHOLD = 0.05
HIGH_THRESHOLD = 0.20
CRITICAL_THRESHOLD = 0.50


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
# HEADER
# =============================================================================


def print_header():

    print()
    print("=" * 80)
    print("SOALM — FINAL SURAT MULTI-SOURCE DATASET")
    print("=" * 80)

    print()
    print("ERA5:")
    print(ERA5_ROOT)

    print()
    print("GDF:")
    print(GDF_FILE)

    print()
    print("Sentinel-1:")
    print(SENTINEL_INDEX)

    print()
    print("OUTPUT:")
    print(OUTPUT_FILE)

    print()
    print("=" * 80)


# =============================================================================
# LOAD ERA5
# =============================================================================


def load_era5():

    print()
    print("=" * 80)
    print("LOADING ERA5")
    print("=" * 80)

    if not ERA5_ROOT.exists():

        raise FileNotFoundError(f"ERA5 directory not found:\n{ERA5_ROOT}")

    files = sorted(ERA5_ROOT.glob("*.parquet"))

    if not files:

        raise FileNotFoundError(f"No ERA5 parquet files found:\n{ERA5_ROOT}")

    print()
    print("ERA5 files found:")
    print(len(files))

    frames = []

    for file in files:

        print(f"  Loading {file.name}")

        df = pd.read_parquet(file)

        if "timestamp" not in df.columns:

            print(f"  ⚠ Skipping {file.name}: " "timestamp missing")

            continue

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True,
        )

        df = df.dropna(subset=["timestamp"])

        frames.append(df)

    if not frames:

        raise RuntimeError("No valid ERA5 datasets found.")

    era5 = pd.concat(
        frames,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Remove duplicate timestamps
    # -------------------------------------------------------------------------

    era5 = (
        era5.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Validate features
    # -------------------------------------------------------------------------

    missing = [feature for feature in FEATURES if feature not in era5.columns]

    if missing:

        raise RuntimeError("Missing SOALM ERA5 features:\n" + "\n".join(missing))

    print()
    print("✓ ERA5 loaded")

    print()
    print(
        "Rows:",
        f"{len(era5):,}",
    )

    print()
    print(
        "Period:",
        era5["timestamp"].min(),
        "→",
        era5["timestamp"].max(),
    )

    return era5


# =============================================================================
# LOAD GDF
# =============================================================================


def load_gdf():

    print()
    print("=" * 80)
    print("LOADING GDF")
    print("=" * 80)

    if not GDF_FILE.exists():

        raise FileNotFoundError(f"GDF file not found:\n{GDF_FILE}")

    gdf = pd.read_csv(GDF_FILE)

    required = [
        "event_time",
        "system_index",
        "flood_fraction",
        "flooded_area_m2",
        "id",
        "max_flood_duration",
        "mean_clear_percentage",
        "valid_area_m2",
        "flood_occurred",
    ]

    missing = [c for c in required if c not in gdf.columns]

    if missing:

        raise RuntimeError("Missing GDF columns:\n" + "\n".join(missing))

    # -------------------------------------------------------------------------
    # event_time
    # -------------------------------------------------------------------------

    numeric_time = pd.to_numeric(
        gdf["event_time"],
        errors="coerce",
    )

    gdf["event_time"] = pd.to_datetime(
        numeric_time,
        unit="ms",
        errors="coerce",
        utc=True,
    )

    # -------------------------------------------------------------------------
    # Numeric columns
    # -------------------------------------------------------------------------

    numeric_columns = [
        "flood_fraction",
        "flooded_area_m2",
        "id",
        "max_flood_duration",
        "mean_clear_percentage",
        "valid_area_m2",
        "flood_occurred",
    ]

    for column in numeric_columns:

        gdf[column] = pd.to_numeric(
            gdf[column],
            errors="coerce",
        )

    gdf["flood_occurred"] = gdf["flood_occurred"].fillna(0).astype("int8")

    gdf = gdf.dropna(subset=["event_time"]).copy()

    print()
    print("GDF rows:")
    print(f"{len(gdf):,}")

    return gdf


# =============================================================================
# PARSE GDF EVENT INTERVAL
# =============================================================================


def parse_event_interval(system_index):

    if pd.isna(system_index):

        return pd.NaT, pd.NaT

    text = str(system_index)

    pattern = r"From_(\d{8})_to_(\d{8})"

    match = re.search(
        pattern,
        text,
    )

    if not match:

        return pd.NaT, pd.NaT

    try:

        start = pd.Timestamp(
            match.group(1),
            tz="UTC",
        )

        end = (
            pd.Timestamp(
                match.group(2),
                tz="UTC",
            )
            + pd.Timedelta(days=1)
            - pd.Timedelta(microseconds=1)
        )

        return start, end

    except Exception:

        return pd.NaT, pd.NaT


# =============================================================================
# BUILD GDF EVENTS
# =============================================================================


def build_gdf_events(gdf):

    print()
    print("=" * 80)
    print("BUILDING GDF EVENT INTERVALS")
    print("=" * 80)

    events = []

    for _, row in gdf.iterrows():

        start, end = parse_event_interval(row["system_index"])

        if pd.isna(start) or pd.isna(end):

            continue

        events.append(
            {
                "gdf_event_id": row["id"],
                "gdf_system_index": row["system_index"],
                "gdf_event_time": row["event_time"],
                "gdf_event_start": start,
                "gdf_event_end": end,
                "gdf_flood_fraction": row["flood_fraction"],
                "gdf_flooded_area_m2": row["flooded_area_m2"],
                "gdf_max_flood_duration": row["max_flood_duration"],
                "gdf_mean_clear_percentage": row["mean_clear_percentage"],
                "gdf_valid_area_m2": row["valid_area_m2"],
                "gdf_flood_occurred": row["flood_occurred"],
            }
        )

    events = pd.DataFrame(events)

    if events.empty:

        raise RuntimeError("No valid GDF events found.")

    events = events.sort_values("gdf_event_start").reset_index(drop=True)

    print()
    print(
        "Valid GDF events:",
        f"{len(events):,}",
    )

    return events


# =============================================================================
# APPLY GDF TO ERA5
# =============================================================================


def merge_gdf(era5, events):
    """
    Merge GDF flood-event intervals with ERA5.

    build_gdf_events() produces:
        gdf_event_start
        gdf_event_end
        gdf_event_time
        gdf_event_id
        gdf_system_index
        gdf_flood_fraction
        gdf_flooded_area_m2
        gdf_max_flood_duration
        gdf_mean_clear_percentage
        gdf_valid_area_m2
        gdf_flood_occurred

    All timestamps are normalized to UTC.
    """

    print()
    print("=" * 80)
    print("MERGING GDF FLOOD EVENTS WITH ERA5")
    print("=" * 80)

    result = era5.copy()
    events = events.copy()

    # =========================================================================
    # DEBUG
    # =========================================================================

    print()
    print("GDF interval columns:")
    print(list(events.columns))

    # =========================================================================
    # CHECK REQUIRED COLUMNS
    # =========================================================================

    required_event_columns = [
        "gdf_event_start",
        "gdf_event_end",
        "gdf_event_time",
        "gdf_event_id",
        "gdf_system_index",
        "gdf_flood_fraction",
        "gdf_flooded_area_m2",
        "gdf_max_flood_duration",
        "gdf_mean_clear_percentage",
        "gdf_valid_area_m2",
        "gdf_flood_occurred",
    ]

    missing = [c for c in required_event_columns if c not in events.columns]

    if missing:
        raise RuntimeError(
            "GDF interval dataset is missing required columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(events.columns)
        )

    # =========================================================================
    # NORMALIZE ERA5 TIMESTAMP
    # =========================================================================

    print()
    print("Normalizing ERA5 timestamps to UTC...")

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
        utc=True,
    )

    result = result.dropna(subset=["timestamp"]).copy()

    result = (
        result.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    # =========================================================================
    # NORMALIZE GDF TIMESTAMPS
    # =========================================================================

    print("Normalizing GDF timestamps to UTC...")

    for column in [
        "gdf_event_start",
        "gdf_event_end",
        "gdf_event_time",
    ]:

        events[column] = pd.to_datetime(
            events[column],
            errors="coerce",
            utc=True,
        )

    # =========================================================================
    # REMOVE INVALID EVENTS
    # =========================================================================

    events = events.loc[
        events["gdf_event_start"].notna() & events["gdf_event_end"].notna()
    ].copy()

    # =========================================================================
    # SORT
    # =========================================================================

    result = result.sort_values("timestamp").reset_index(drop=True)

    events = events.sort_values("gdf_event_start").reset_index(drop=True)

    # =========================================================================
    # ERA5 RANGE
    # =========================================================================

    era5_start = result["timestamp"].min()
    era5_end = result["timestamp"].max()

    print()
    print("ERA5 period:")
    print(f"  {era5_start}")
    print(f"  {era5_end}")

    print()
    print(f"ERA5 rows: {len(result):,}")

    # =========================================================================
    # FILTER GDF EVENTS THAT OVERLAP ERA5
    # =========================================================================

    events = events.loc[
        (events["gdf_event_end"] >= era5_start)
        & (events["gdf_event_start"] <= era5_end)
    ].copy()

    print()
    print(
        "GDF events overlapping ERA5:",
        f"{len(events):,}",
    )

    # =========================================================================
    # INITIALIZE GDF COLUMNS
    # =========================================================================

    result["flood_occurred"] = np.int8(0)

    result["gdf_event_time"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns, UTC]",
    )

    result["gdf_event_start"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns, UTC]",
    )

    result["gdf_event_end"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns, UTC]",
    )

    result["gdf_event_id"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    result["gdf_system_index"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="object",
    )

    result["gdf_flood_fraction"] = 0.0
    result["gdf_flooded_area_m2"] = 0.0
    result["gdf_max_flood_duration"] = 0.0
    result["gdf_mean_clear_percentage"] = np.nan
    result["gdf_valid_area_m2"] = 0.0

    # =========================================================================
    # NO EVENTS
    # =========================================================================

    if events.empty:

        print()
        print("No GDF events overlap the ERA5 period.")

        return result

    # =========================================================================
    # MATCH EACH GDF EVENT TO ERA5
    # =========================================================================

    total_matched = 0

    for event_number, (_, event) in enumerate(
        events.iterrows(),
        start=1,
    ):

        # ---------------------------------------------------------------------
        # EVENT START
        # ---------------------------------------------------------------------

        event_start = pd.Timestamp(event["gdf_event_start"])

        if event_start.tzinfo is None:
            event_start = event_start.tz_localize("UTC")
        else:
            event_start = event_start.tz_convert("UTC")

        # ---------------------------------------------------------------------
        # EVENT END
        # ---------------------------------------------------------------------

        event_end = pd.Timestamp(event["gdf_event_end"])

        if event_end.tzinfo is None:
            event_end = event_end.tz_localize("UTC")
        else:
            event_end = event_end.tz_convert("UTC")

        # ---------------------------------------------------------------------
        # CLAMP EVENT TO ERA5 RANGE
        # ---------------------------------------------------------------------

        match_start = max(
            event_start,
            era5_start,
        )

        match_end = min(
            event_end,
            era5_end,
        )

        if match_start > match_end:
            continue

        # ---------------------------------------------------------------------
        # MATCH HOURLY ERA5 ROWS
        # ---------------------------------------------------------------------

        mask = (result["timestamp"] >= match_start) & (result["timestamp"] <= match_end)

        count = int(mask.sum())

        # ---------------------------------------------------------------------
        # FALLBACK NEAREST ERA5 ROW
        # ---------------------------------------------------------------------

        if count == 0:

            difference = (result["timestamp"] - event_start).abs()

            nearest_index = difference.idxmin()

            nearest_difference = difference.loc[nearest_index]

            if nearest_difference <= pd.Timedelta(hours=12):

                mask = pd.Series(
                    False,
                    index=result.index,
                )

                mask.loc[nearest_index] = True

                count = 1

        if count == 0:
            continue

        # =========================================================================
        # FLOOD LABEL
        # =========================================================================

        result.loc[
            mask,
            "flood_occurred",
        ] = 1

        # =========================================================================
        # GDF EVENT TIME
        # =========================================================================

        event_time = event["gdf_event_time"]

        if pd.notna(event_time):

            event_time = pd.Timestamp(event_time)

            if event_time.tzinfo is None:
                event_time = event_time.tz_localize("UTC")
            else:
                event_time = event_time.tz_convert("UTC")

            result.loc[
                mask,
                "gdf_event_time",
            ] = event_time

        # =========================================================================
        # EVENT START / END
        # =========================================================================

        result.loc[
            mask,
            "gdf_event_start",
        ] = event_start

        result.loc[
            mask,
            "gdf_event_end",
        ] = event_end

        # =========================================================================
        # EVENT ID
        # =========================================================================

        event_id = event["gdf_event_id"]

        if pd.notna(event_id):

            try:

                result.loc[
                    mask,
                    "gdf_event_id",
                ] = int(event_id)

            except (
                ValueError,
                TypeError,
            ):
                pass

        # =========================================================================
        # SYSTEM INDEX
        # =========================================================================

        system_index = event["gdf_system_index"]

        if pd.notna(system_index):

            result.loc[
                mask,
                "gdf_system_index",
            ] = str(system_index)

        # =========================================================================
        # GDF NUMERIC FIELDS
        # =========================================================================

        numeric_mapping = {
            "gdf_flood_fraction": "gdf_flood_fraction",
            "gdf_flooded_area_m2": "gdf_flooded_area_m2",
            "gdf_max_flood_duration": "gdf_max_flood_duration",
            "gdf_mean_clear_percentage": "gdf_mean_clear_percentage",
            "gdf_valid_area_m2": "gdf_valid_area_m2",
        }

        for source_column, target_column in numeric_mapping.items():

            value = pd.to_numeric(
                event[source_column],
                errors="coerce",
            )

            if pd.notna(value):

                result.loc[
                    mask,
                    target_column,
                ] = float(value)

        # =========================================================================
        # LOG
        # =========================================================================

        total_matched += count

        print(
            f"  Event {event_number:03d}: "
            f"{event['gdf_system_index']} | "
            f"{event_start.date()} → "
            f"{event_end.date()} | "
            f"{count:,} ERA5 rows"
        )

    # =========================================================================
    # FINAL STATISTICS
    # =========================================================================

    flood_rows = int(result["flood_occurred"].sum())

    print()
    print("=" * 80)
    print("GDF ↔ ERA5 MERGE COMPLETE")
    print("=" * 80)

    print()
    print(
        "ERA5 rows:",
        f"{len(result):,}",
    )

    print(
        "GDF events used:",
        f"{len(events):,}",
    )

    print(
        "Matched ERA5 rows:",
        f"{total_matched:,}",
    )

    print(
        "Flood-labelled ERA5 rows:",
        f"{flood_rows:,}",
    )

    print(
        "Non-flood ERA5 rows:",
        f"{len(result) - flood_rows:,}",
    )

    if len(result) > 0:

        print("Flood-labelled percentage:", f"{flood_rows / len(result) * 100:.4f}%")

    return result


# =============================================================================
# LOAD SENTINEL
# =============================================================================


def load_sentinel():

    print()
    print("=" * 80)
    print("LOADING SENTINEL-1 INDEX")
    print("=" * 80)

    if not SENTINEL_INDEX.exists():

        print()
        print("⚠ Sentinel-1 index not found.")

        print(SENTINEL_INDEX)

        print()
        print("Continuing without Sentinel-1.")

        return pd.DataFrame()

    sentinel = pd.read_parquet(SENTINEL_INDEX)

    if sentinel.empty:

        print("⚠ Sentinel-1 index is empty.")

        return sentinel

    # -------------------------------------------------------------------------
    # Normalize timestamp
    # -------------------------------------------------------------------------

    sentinel["sentinel_datetime"] = pd.to_datetime(
        sentinel["sentinel_datetime"],
        errors="coerce",
        utc=True,
    )

    sentinel = sentinel.dropna(subset=["sentinel_datetime"]).copy()

    sentinel = (
        sentinel.sort_values("sentinel_datetime")
        .drop_duplicates(
            subset=["product_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print()
    print(
        "Sentinel-1 products:",
        f"{len(sentinel):,}",
    )

    print()
    print(
        "Sentinel-1 period:",
        sentinel["sentinel_datetime"].min(),
        "→",
        sentinel["sentinel_datetime"].max(),
    )

    return sentinel


# =============================================================================
# MERGE SENTINEL
# =============================================================================

# =============================================================================
# MERGE SENTINEL
# =============================================================================


def merge_sentinel(
    era5_gdf,
    sentinel,
):
    """
    Merge Sentinel-1 acquisition metadata with ERA5 + GDF data.

    One row remains one ERA5 timestamp.

    Sentinel-1 acquisitions are matched to the nearest ERA5 timestamp
    within MAX_SENTINEL_DISTANCE_HOURS.

    IMPORTANT:
    Sentinel-1 metadata is NOT SAR backscatter.
    No VV/VH values are being fabricated.
    """

    print()
    print("=" * 80)
    print("MERGING SENTINEL-1")
    print("=" * 80)

    result = era5_gdf.copy()

    # =========================================================================
    # NO SENTINEL DATA
    # =========================================================================

    if sentinel is None or sentinel.empty:

        print()
        print("No Sentinel-1 products available.")

        # Always create final Sentinel columns
        result["sentinel_available"] = np.int8(0)
        result["sentinel_datetime"] = pd.NaT
        result["sentinel_product_id"] = None
        result["sentinel_platform"] = None
        result["sentinel_orbit_direction"] = None
        result["sentinel_relative_orbit"] = np.nan
        result["sentinel_polarization"] = None
        result["sentinel_product_type"] = None
        result["sentinel_cloud_cover"] = np.nan
        result["sentinel_time_difference_hours"] = np.nan
        result["sentinel_event_match"] = np.int8(0)

        return result

    # =========================================================================
    # COPY
    # =========================================================================

    sentinel = sentinel.copy()

    # =========================================================================
    # VALIDATE SENTINEL TIMESTAMP
    # =========================================================================

    if "sentinel_datetime" not in sentinel.columns:

        raise RuntimeError("Sentinel-1 index does not contain 'sentinel_datetime'.")

    # =========================================================================
    # NORMALIZE ERA5 TIMESTAMP
    # =========================================================================

    print()
    print("Normalizing ERA5 timestamps...")

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
        utc=True,
    )

    result = result.dropna(subset=["timestamp"]).copy()

    # Force nanosecond precision
    result["timestamp"] = result["timestamp"].astype("datetime64[ns, UTC]")

    result = result.sort_values("timestamp").reset_index(drop=True)

    # =========================================================================
    # NORMALIZE SENTINEL TIMESTAMP
    # =========================================================================

    print("Normalizing Sentinel-1 timestamps...")

    sentinel["sentinel_datetime"] = pd.to_datetime(
        sentinel["sentinel_datetime"],
        errors="coerce",
        utc=True,
    )

    sentinel = sentinel.dropna(subset=["sentinel_datetime"]).copy()

    sentinel["sentinel_datetime"] = sentinel["sentinel_datetime"].astype(
        "datetime64[ns, UTC]"
    )

    # =========================================================================
    # OPTIONAL SENTINEL COLUMNS
    # =========================================================================

    optional_columns = [
        "product_id",
        "platform",
        "orbit_direction",
        "relative_orbit",
        "polarization",
        "product_type",
        "cloud_cover",
    ]

    for column in optional_columns:

        if column not in sentinel.columns:
            sentinel[column] = None

    # =========================================================================
    # REMOVE DUPLICATE PRODUCTS
    # =========================================================================

    sentinel = (
        sentinel.sort_values("sentinel_datetime")
        .drop_duplicates(
            subset=["product_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    # =========================================================================
    # DEBUG
    # =========================================================================

    print()
    print("Timestamp dtypes before merge:")

    print(
        "  ERA5:",
        result["timestamp"].dtype,
    )

    print(
        "  Sentinel:",
        sentinel["sentinel_datetime"].dtype,
    )

    # =========================================================================
    # PREPARE SENTINEL DATA
    # =========================================================================

    sentinel_for_merge = sentinel[
        [
            "sentinel_datetime",
            "product_id",
            "platform",
            "orbit_direction",
            "relative_orbit",
            "polarization",
            "product_type",
            "cloud_cover",
        ]
    ].copy()

    sentinel_for_merge = sentinel_for_merge.sort_values(
        "sentinel_datetime"
    ).reset_index(drop=True)

    # =========================================================================
    # FINAL TIMESTAMP NORMALIZATION
    # =========================================================================

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
    ).astype("datetime64[ns, UTC]")

    sentinel_for_merge["sentinel_datetime"] = pd.to_datetime(
        sentinel_for_merge["sentinel_datetime"],
        utc=True,
    ).astype("datetime64[ns, UTC]")

    print()
    print("Final merge key dtypes:")

    print(
        "  ERA5 timestamp:",
        result["timestamp"].dtype,
    )

    print(
        "  Sentinel match time:",
        sentinel_for_merge["sentinel_datetime"].dtype,
    )

    # =========================================================================
    # IMPORTANT:
    # REMOVE ANY OLD SENTINEL PLACEHOLDER COLUMNS
    # =========================================================================
    #
    # This is the key fix.
    #
    # merge_asof() will create the Sentinel fields itself.
    #
    # We must NOT already have columns such as
    # sentinel_product_id on the left side.

    old_sentinel_columns = [
        "sentinel_available",
        "sentinel_datetime",
        "sentinel_product_id",
        "sentinel_platform",
        "sentinel_orbit_direction",
        "sentinel_relative_orbit",
        "sentinel_polarization",
        "sentinel_product_type",
        "sentinel_cloud_cover",
        "sentinel_time_difference_hours",
        "sentinel_event_match",
    ]

    columns_to_drop = [
        column for column in old_sentinel_columns if column in result.columns
    ]

    if columns_to_drop:

        result = result.drop(columns=columns_to_drop)

    # =========================================================================
    # MERGE ASOF
    # =========================================================================

    print()
    print(
        "Matching Sentinel-1 acquisitions within",
        MAX_SENTINEL_DISTANCE_HOURS,
        "hours...",
    )

    merged = pd.merge_asof(
        result.sort_values("timestamp"),
        sentinel_for_merge.sort_values("sentinel_datetime"),
        left_on="timestamp",
        right_on="sentinel_datetime",
        direction="nearest",
        tolerance=pd.Timedelta(hours=MAX_SENTINEL_DISTANCE_HOURS),
    )

    # =========================================================================
    # RENAME SENTINEL FIELDS
    # =========================================================================

    merged = merged.rename(
        columns={
            "product_id": "sentinel_product_id",
            "platform": "sentinel_platform",
            "orbit_direction": "sentinel_orbit_direction",
            "relative_orbit": "sentinel_relative_orbit",
            "polarization": "sentinel_polarization",
            "product_type": "sentinel_product_type",
            "cloud_cover": "sentinel_cloud_cover",
        }
    )

    # =========================================================================
    # SENTINEL DATETIME
    # =========================================================================

    merged["sentinel_datetime"] = pd.to_datetime(
        merged["sentinel_datetime"],
        errors="coerce",
        utc=True,
    ).astype("datetime64[ns, UTC]")

    # =========================================================================
    # AVAILABILITY
    # =========================================================================

    merged["sentinel_available"] = merged["sentinel_datetime"].notna().astype("int8")

    # =========================================================================
    # TIME DIFFERENCE
    # =========================================================================

    merged["sentinel_time_difference_hours"] = np.nan

    valid_sentinel = merged["sentinel_datetime"].notna()

    merged.loc[
        valid_sentinel,
        "sentinel_time_difference_hours",
    ] = (
        merged.loc[
            valid_sentinel,
            "timestamp",
        ]
        - merged.loc[
            valid_sentinel,
            "sentinel_datetime",
        ]
    ).abs().dt.total_seconds() / 3600.0

    # =========================================================================
    # SENTINEL INSIDE GDF EVENT
    # =========================================================================

    merged["sentinel_event_match"] = (
        merged["sentinel_datetime"].notna()
        & merged["gdf_event_start"].notna()
        & merged["gdf_event_end"].notna()
        & (merged["sentinel_datetime"] >= merged["gdf_event_start"])
        & (merged["sentinel_datetime"] <= merged["gdf_event_end"])
    ).astype("int8")

    # =========================================================================
    # ENSURE SATELLITE COLUMNS HAVE EXPECTED TYPES
    # =========================================================================

    if "sentinel_relative_orbit" in merged.columns:

        merged["sentinel_relative_orbit"] = pd.to_numeric(
            merged["sentinel_relative_orbit"],
            errors="coerce",
        )

    if "sentinel_cloud_cover" in merged.columns:

        merged["sentinel_cloud_cover"] = pd.to_numeric(
            merged["sentinel_cloud_cover"],
            errors="coerce",
        )

    # =========================================================================
    # SATELLITE OBSERVATION SIGNAL
    # =========================================================================

    merged["satellite_observation_available"] = merged["sentinel_available"]

    # =========================================================================
    # REMOVE ANY ACCIDENTAL DUPLICATE COLUMNS
    # =========================================================================

    if merged.columns.duplicated().any():

        duplicates = merged.columns[merged.columns.duplicated()].tolist()

        raise RuntimeError(
            "Duplicate columns detected after Sentinel merge:\n"
            + "\n".join(map(str, duplicates))
        )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    print()
    print("=" * 80)
    print("SENTINEL-1 MERGE COMPLETE")
    print("=" * 80)

    print()
    print(
        "ERA5 rows:",
        f"{len(merged):,}",
    )

    sentinel_rows = int(merged["sentinel_available"].sum())

    print(
        "Rows with Sentinel-1:",
        f"{sentinel_rows:,}",
    )

    if len(merged) > 0:

        print(
            "Sentinel availability:",
            f"{merged['sentinel_available'].mean() * 100:.3f}%",
        )

    print()
    print(
        "Sentinel acquisitions during GDF events:",
        int(merged["sentinel_event_match"].sum()),
    )

    print()
    print("Sentinel columns after merge:")

    sentinel_output_columns = [
        "sentinel_available",
        "sentinel_datetime",
        "sentinel_product_id",
        "sentinel_platform",
        "sentinel_orbit_direction",
        "sentinel_relative_orbit",
        "sentinel_polarization",
        "sentinel_product_type",
        "sentinel_cloud_cover",
        "sentinel_time_difference_hours",
        "sentinel_event_match",
    ]

    print([c for c in sentinel_output_columns if c in merged.columns])

    return merged


# =============================================================================
# DERIVE FLOOD SEVERITY
# =============================================================================


def create_flood_severity(
    df,
):

    print()
    print("=" * 80)
    print("CREATING FLOOD SEVERITY CLASSES")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # IMPORTANT
    # -------------------------------------------------------------------------
    #
    # We only assign severity when flood_occurred == 1.
    #
    # Non-flood:
    #
    #     0 = NONE
    #
    # Flood:
    #
    #     1 = LOW
    #     2 = MODERATE
    #     3 = HIGH
    #     4 = CRITICAL
    #
    # -------------------------------------------------------------------------

    df["flood_severity"] = np.int8(0)

    df["flood_class"] = "NONE"

    flood_mask = df["flood_occurred"] == 1

    fraction = pd.to_numeric(
        df["gdf_flood_fraction"],
        errors="coerce",
    ).fillna(0.0)

    # -------------------------------------------------------------------------
    # LOW
    # -------------------------------------------------------------------------

    low_mask = flood_mask & (fraction > LOW_THRESHOLD) & (fraction < MODERATE_THRESHOLD)

    df.loc[low_mask, "flood_severity"] = 1

    df.loc[low_mask, "flood_class"] = "LOW"

    # -------------------------------------------------------------------------
    # MODERATE
    # -------------------------------------------------------------------------

    moderate_mask = (
        flood_mask & (fraction >= MODERATE_THRESHOLD) & (fraction < HIGH_THRESHOLD)
    )

    df.loc[moderate_mask, "flood_severity"] = 2

    df.loc[moderate_mask, "flood_class"] = "MODERATE"

    # -------------------------------------------------------------------------
    # HIGH
    # -------------------------------------------------------------------------

    high_mask = (
        flood_mask & (fraction >= HIGH_THRESHOLD) & (fraction < CRITICAL_THRESHOLD)
    )

    df.loc[high_mask, "flood_severity"] = 3

    df.loc[high_mask, "flood_class"] = "HIGH"

    # -------------------------------------------------------------------------
    # CRITICAL
    # -------------------------------------------------------------------------

    critical_mask = flood_mask & (fraction >= CRITICAL_THRESHOLD)

    df.loc[critical_mask, "flood_severity"] = 4

    df.loc[critical_mask, "flood_class"] = "CRITICAL"

    print()
    print(df["flood_class"].value_counts().to_string())

    return df


# =============================================================================
# DERIVED FEATURES
# =============================================================================


def create_derived_features(
    df,
):

    print()
    print("=" * 80)
    print("CREATING DERIVED FEATURES")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Rainfall intensity indicators
    # -------------------------------------------------------------------------

    df["rain_accumulation_ratio_24_72"] = df["rain_last_24h"] / (
        df["rain_last_72h"] + 1e-6
    )

    df["rain_accumulation_ratio_6_24"] = df["rain_last_6h"] / (
        df["rain_last_24h"] + 1e-6
    )

    # -------------------------------------------------------------------------
    # Pressure stress
    # -------------------------------------------------------------------------

    df["pressure_drop_6h"] = -df["pressure_change_6h"]

    # -------------------------------------------------------------------------
    # Combined rainfall signal
    # -------------------------------------------------------------------------

    df["extreme_rain_signal"] = (
        (df["rain_last_24h"] >= 60) | (df["rain_last_72h"] >= 100)
    ).astype("int8")

    # -------------------------------------------------------------------------
    # Sentinel availability signal
    # -------------------------------------------------------------------------

    df["satellite_observation_available"] = df["sentinel_available"]

    print()
    print("Derived features created.")

    return df


# =============================================================================
# ARRANGE FINAL COLUMNS
# =============================================================================


def arrange_columns(
    df,
):

    base = [
        "timestamp",
        "city",
    ]

    gdf_columns = [
        "flood_occurred",
        "flood_severity",
        "flood_class",
        "gdf_event_id",
        "gdf_system_index",
        "gdf_event_time",
        "gdf_event_start",
        "gdf_event_end",
        "gdf_flood_fraction",
        "gdf_flooded_area_m2",
        "gdf_max_flood_duration",
        "gdf_mean_clear_percentage",
        "gdf_valid_area_m2",
    ]

    sentinel_columns = [
        "sentinel_available",
        "satellite_observation_available",
        "sentinel_event_match",
        "sentinel_datetime",
        "sentinel_time_difference_hours",
        "sentinel_product_id",
        "sentinel_platform",
        "sentinel_orbit_direction",
        "sentinel_relative_orbit",
        "sentinel_polarization",
        "sentinel_product_type",
        "sentinel_cloud_cover",
    ]

    derived_columns = [
        "rain_accumulation_ratio_24_72",
        "rain_accumulation_ratio_6_24",
        "pressure_drop_6h",
        "extreme_rain_signal",
    ]

    columns = base + FEATURES + gdf_columns + sentinel_columns + derived_columns

    columns = [column for column in columns if column in df.columns]

    return df[columns].copy()


# =============================================================================
# VALIDATION
# =============================================================================


def validate(
    df,
):

    print()
    print("=" * 80)
    print("FINAL DATASET VALIDATION")
    print("=" * 80)

    print()
    print(
        "Rows:",
        f"{len(df):,}",
    )

    print()
    print(
        "Columns:",
        len(df.columns),
    )

    print()
    print(
        "SOALM ERA5 features:",
        sum(f in df.columns for f in FEATURES),
        "/ 40",
    )

    print()
    print(
        "Flood rows:",
        int(df["flood_occurred"].sum()),
    )

    print()
    print("Flood classes:")

    print(df["flood_class"].value_counts().to_string())

    print()
    print("Sentinel-1:")

    print(
        "Rows with Sentinel:",
        int(df["sentinel_available"].sum()),
    )

    print(
        "Sentinel event matches:",
        int(df["sentinel_event_match"].sum()),
    )

    print()
    print("Timestamp:")

    print(
        df["timestamp"].min(),
        "→",
        df["timestamp"].max(),
    )

    # -------------------------------------------------------------------------
    # Check duplicate timestamps
    # -------------------------------------------------------------------------

    duplicates = int(df["timestamp"].duplicated().sum())

    print()
    print(
        "Duplicate timestamps:",
        duplicates,
    )

    if duplicates:

        print("⚠ Duplicate timestamps detected.")

    # -------------------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------------------

    print()
    print("Major missing values:")

    missing = df.isna().sum().sort_values(ascending=False).head(15)

    print(missing.to_string())

    # -------------------------------------------------------------------------
    # Flood examples
    # -------------------------------------------------------------------------

    flood = df[df["flood_occurred"] == 1]

    if not flood.empty:

        print()
        print("Sample flood rows:")

        columns = [
            "timestamp",
            "flood_occurred",
            "flood_class",
            "gdf_flood_fraction",
            "gdf_flooded_area_m2",
            "sentinel_available",
            "sentinel_datetime",
            "sentinel_product_id",
        ]

        columns = [c for c in columns if c in flood.columns]

        print(flood[columns].head(15).to_string(index=False))


# =============================================================================
# SAVE
# =============================================================================


def save(
    df,
):

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("SAVING FINAL DATASET")
    print("=" * 80)

    df.to_parquet(
        OUTPUT_FILE,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    size_mb = OUTPUT_FILE.stat().st_size / (1024**2)

    print()
    print("✓ Saved successfully")

    print()
    print("File:")

    print(OUTPUT_FILE)

    print()
    print(
        "Rows:",
        f"{len(df):,}",
    )

    print(
        "Columns:",
        len(df.columns),
    )

    print(
        "Size:",
        f"{size_mb:.2f} MB",
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    print_header()

    try:

        # =====================================================================
        # 1. ERA5
        # =====================================================================

        era5 = load_era5()

        # =====================================================================
        # 2. GDF
        # =====================================================================

        gdf = load_gdf()

        # =====================================================================
        # 3. Build flood intervals
        # =====================================================================

        events = build_gdf_events(gdf)

        # =====================================================================
        # 4. Merge GDF + ERA5
        # =====================================================================

        combined = merge_gdf(
            era5,
            events,
        )

        # =====================================================================
        # 5. Sentinel
        # =====================================================================

        sentinel = load_sentinel()

        combined = merge_sentinel(
            combined,
            sentinel,
        )

        # =====================================================================
        # 6. Flood severity
        # =====================================================================

        combined = create_flood_severity(combined)

        # =====================================================================
        # 7. Derived features
        # =====================================================================

        combined = create_derived_features(combined)

        # =====================================================================
        # 8. Arrange
        # =====================================================================

        combined = arrange_columns(combined)

        # =====================================================================
        # 9. Validate
        # =====================================================================

        validate(combined)

        # =====================================================================
        # 10. Save
        # =====================================================================

        save(combined)

        print()
        print("=" * 80)
        print("COMPLETE")
        print("=" * 80)

        print()
        print("FINAL SOALM DATASET:")

        print(OUTPUT_FILE)

        print()
        print("ERA5 + GDF + Sentinel-1 metadata successfully combined.")

        print()
        print("IMPORTANT:")

        print(
            "Current Sentinel-1 columns represent satellite " "observations/metadata."
        )

        print("Actual VV/VH SAR backscatter features are NOT yet " "included.")

        print()
        print("Next recommended step:")

        print(
            "Download Sentinel-1 products and extract VV/VH/" "water-extent features."
        )

    except KeyboardInterrupt:

        print()
        print("Interrupted.")

        sys.exit(1)

    except Exception as exc:

        print()
        print("=" * 80)
        print("FATAL ERROR")
        print("=" * 80)

        print()
        print(repr(exc))

        traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    main()
