"""
===============================================================================
SOALM — SURAT ERA5 + GDF FLOOD DATA PREPARATION
===============================================================================

PURPOSE
-------
Combine ONE YEAR of Surat ERA5 data with Global Flood Database (GDF)
flood-event observations.

IMPORTANT
---------
All timestamps are normalized internally to:

    UTC + timezone-naive datetime64[ns]

This completely avoids pandas timezone-aware / timezone-naive conflicts.

GDF event intervals are converted to hourly ERA5 labels.

Example:

    DFO_2049_From_20020903_to_20020906

becomes:

    2002-09-03 00:00
    through
    2002-09-06 23:00

All ERA5 timestamps inside that interval receive:

    flood_occurred = 1

Original ERA5 files are NEVER modified.

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
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ERA5_FEATURE_ROOT = PROJECT_ROOT / "data" / "era5" / "features"

# IMPORTANT:
# Actual Surat GFD filename
GDF_FILE = (
    PROJECT_ROOT / "data" / "flood_training" / "surat" / "raw" / "gfd" / "surat.csv"
)

PREPARED_ROOT = PROJECT_ROOT / "data" / "era5" / "prepared"


# =============================================================================
# YEAR
# =============================================================================

YEAR = 2005


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_FILE = PREPARED_ROOT / f"surat_{YEAR}_gdf_era5.parquet"


# =============================================================================
# OVERWRITE
# =============================================================================

OVERWRITE = True


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
# GDF COLUMNS
# =============================================================================

GDF_COLUMNS = [
    "event_time",
    "flood_fraction",
    "flooded_area_m2",
    "id",
    "max_flood_duration",
    "mean_clear_percentage",
    "system_index",
    "valid_area_m2",
    "flood_occurred",
]


# =============================================================================
# TIMESTAMP NORMALIZATION
# =============================================================================


def normalize_timestamp_utc_naive(value):
    """
    Convert arbitrary timestamp input to UTC timezone-naive Timestamp.

    Final representation:

        Timestamp('2005-06-28 00:00:00')

    No timezone information remains.

    This is deliberate.

    We use UTC as the common reference and remove timezone information
    only after converting to UTC.
    """

    if value is None:
        return pd.NaT

    try:

        # ---------------------------------------------------------------------
        # Already pandas Timestamp
        # ---------------------------------------------------------------------

        if isinstance(value, pd.Timestamp):

            if pd.isna(value):
                return pd.NaT

            if value.tzinfo is not None:
                return value.tz_convert("UTC").tz_localize(None)

            return value

        # ---------------------------------------------------------------------
        # Python datetime
        # ---------------------------------------------------------------------

        parsed = pd.to_datetime(
            value,
            errors="coerce",
            utc=True,
        )

        if pd.isna(parsed):
            return pd.NaT

        return pd.Timestamp(parsed).tz_localize(None)

    except Exception:

        return pd.NaT


# =============================================================================
# NORMALIZE SERIES
# =============================================================================


def normalize_timestamp_series(series):
    """
    Convert a complete timestamp Series to UTC-naive datetime64[ns].

    IMPORTANT:

    We construct a NEW Series instead of assigning a timezone-aware
    DatetimeArray into an existing datetime64[ns] column.

    This avoids the pandas error:

        Invalid value '<DatetimeArray ... UTC>'
        for dtype 'datetime64[ns]'
    """

    # -------------------------------------------------------------------------
    # First attempt: pandas handles normal datetime strings / timestamps.
    # -------------------------------------------------------------------------

    parsed = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    # -------------------------------------------------------------------------
    # Convert UTC-aware -> UTC-naive.
    # -------------------------------------------------------------------------

    try:

        parsed = parsed.dt.tz_localize(None)

    except AttributeError:

        # Scalar-like fallback
        parsed = pd.Series(
            [normalize_timestamp_utc_naive(value) for value in series],
            index=series.index,
        )

    # -------------------------------------------------------------------------
    # Force nanosecond precision.
    # -------------------------------------------------------------------------

    return pd.Series(
        pd.array(
            parsed,
            dtype="datetime64[ns]",
        ),
        index=series.index,
        name=series.name,
    )


# =============================================================================
# HEADER
# =============================================================================


def print_header():

    print()
    print("=" * 80)
    print("SOALM — SURAT ERA5 + GDF FLOOD DATA PREPARATION")
    print("=" * 80)

    print()
    print("City:")
    print("Surat")

    print()
    print("Year:")
    print(YEAR)

    print()
    print("ERA5 source:")
    print(ERA5_FEATURE_ROOT)

    print()
    print("GDF source:")
    print(GDF_FILE)

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print("=" * 80)


# =============================================================================
# FIND ERA5
# =============================================================================


def find_era5_file():

    city_dir = ERA5_FEATURE_ROOT / "surat"

    if not city_dir.exists():

        raise FileNotFoundError(f"\nSurat ERA5 directory not found:\n{city_dir}\n")

    exact_file = city_dir / f"{YEAR}.parquet"

    if exact_file.exists():
        return exact_file

    candidates = sorted(city_dir.glob(f"*{YEAR}*.parquet"))

    if not candidates:

        raise FileNotFoundError(
            "\nCould not find Surat ERA5 file for year "
            f"{YEAR}.\n\n"
            f"Expected:\n{exact_file}\n"
        )

    return candidates[0]


# =============================================================================
# LOAD ERA5
# =============================================================================


def load_era5():

    path = find_era5_file()

    print()
    print("=" * 80)
    print("LOADING SURAT ERA5")
    print("=" * 80)

    print()
    print("File:")
    print(path)

    df = pd.read_parquet(path)

    print()
    print("Rows:")
    print(f"{len(df):,}")

    print()
    print("Columns:")
    print(len(df.columns))

    # -------------------------------------------------------------------------
    # Required timestamp
    # -------------------------------------------------------------------------

    if "timestamp" not in df.columns:

        raise RuntimeError("ERA5 dataset does not contain 'timestamp'.")

    # -------------------------------------------------------------------------
    # Normalize timestamp
    # -------------------------------------------------------------------------

    print()
    print("Normalizing ERA5 timestamps to UTC...")

    df["timestamp"] = normalize_timestamp_series(df["timestamp"])

    df = df.dropna(subset=["timestamp"]).copy()

    # -------------------------------------------------------------------------
    # Requested year
    # -------------------------------------------------------------------------

    df = df[df["timestamp"].dt.year == YEAR].copy()

    if df.empty:

        raise RuntimeError(f"No ERA5 rows found for Surat {YEAR}.")

    # -------------------------------------------------------------------------
    # Features
    # -------------------------------------------------------------------------

    missing = [column for column in FEATURES if column not in df.columns]

    if missing:

        raise RuntimeError("Missing ERA5 features:\n" + "\n".join(missing))

    # -------------------------------------------------------------------------
    # Sort and deduplicate
    # -------------------------------------------------------------------------

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print()
    print("✓ ERA5 loaded")

    print(
        "ERA5 period:",
        df["timestamp"].min(),
        "→",
        df["timestamp"].max(),
    )

    print(
        "ERA5 rows:",
        f"{len(df):,}",
    )

    print(
        "ERA5 timestamp dtype:",
        df["timestamp"].dtype,
    )

    return df


# =============================================================================
# PARSE NUMERIC GDF EVENT TIME
# =============================================================================


def parse_gdf_event_time_value(value):
    """
    Robustly parse GDF event_time.

    GDF datasets commonly store event_time as Unix milliseconds.

    We support:

        Unix milliseconds
        Unix seconds
        datetime strings
        pandas timestamps

    Final result is ALWAYS UTC-naive.
    """

    if pd.isna(value):
        return pd.NaT

    # -------------------------------------------------------------------------
    # Numeric timestamp
    # -------------------------------------------------------------------------

    try:

        numeric = float(value)

        if np.isfinite(numeric):

            # Unix milliseconds
            if abs(numeric) > 1e11:

                ts = pd.to_datetime(
                    numeric,
                    unit="ms",
                    errors="coerce",
                    utc=True,
                )

            # Unix seconds
            elif abs(numeric) > 1e9:

                ts = pd.to_datetime(
                    numeric,
                    unit="s",
                    errors="coerce",
                    utc=True,
                )

            else:

                ts = pd.NaT

            if pd.notna(ts):

                return pd.Timestamp(ts).tz_localize(None)

    except (
        ValueError,
        TypeError,
        OverflowError,
    ):

        pass

    # -------------------------------------------------------------------------
    # Datetime string
    # -------------------------------------------------------------------------

    try:

        ts = pd.to_datetime(
            value,
            errors="coerce",
            utc=True,
        )

        if pd.notna(ts):

            return pd.Timestamp(ts).tz_localize(None)

    except Exception:

        pass

    return pd.NaT


# =============================================================================
# LOAD GDF
# =============================================================================


def load_gdf():

    print()
    print("=" * 80)
    print("LOADING GDF FLOOD DATA")
    print("=" * 80)

    if not GDF_FILE.exists():

        raise FileNotFoundError(f"\nGDF file not found:\n{GDF_FILE}\n")

    print()
    print("File:")
    print(GDF_FILE)

    gdf = pd.read_csv(
        GDF_FILE,
        low_memory=False,
    )

    print()
    print("Rows:")
    print(f"{len(gdf):,}")

    print()
    print("Columns:")
    print(list(gdf.columns))

    # -------------------------------------------------------------------------
    # Required columns
    # -------------------------------------------------------------------------

    missing = [column for column in GDF_COLUMNS if column not in gdf.columns]

    if missing:

        raise RuntimeError("Missing GDF columns:\n" + "\n".join(missing))

    # -------------------------------------------------------------------------
    # EVENT TIME
    #
    # IMPORTANT:
    # Do NOT do:
    #
    # gdf.loc[mask, "event_time"] = aware_datetime_array
    #
    # because that causes the exact pandas error you encountered.
    #
    # Instead create the complete normalized Series first.
    # -------------------------------------------------------------------------

    print()
    print("Normalizing GDF event_time to UTC...")

    raw_event_time = gdf["event_time"].copy()

    normalized_event_time = pd.Series(
        [parse_gdf_event_time_value(value) for value in raw_event_time],
        index=gdf.index,
        dtype="datetime64[ns]",
        name="event_time",
    )

    gdf["event_time"] = normalized_event_time

    # -------------------------------------------------------------------------
    # FLOOD OCCURRED
    # -------------------------------------------------------------------------

    gdf["flood_occurred"] = (
        pd.to_numeric(
            gdf["flood_occurred"],
            errors="coerce",
        )
        .fillna(0)
        .astype("int8")
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
    ]

    for column in numeric_columns:

        gdf[column] = pd.to_numeric(
            gdf[column],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # Remove invalid timestamps
    # -------------------------------------------------------------------------

    before = len(gdf)

    gdf = gdf[gdf["event_time"].notna()].copy()

    print()
    print(
        "Invalid GDF timestamps removed:",
        before - len(gdf),
    )

    print()
    print("✓ GDF loaded")

    print()
    print(
        "GDF event_time dtype:",
        gdf["event_time"].dtype,
    )

    return gdf


# =============================================================================
# PARSE EVENT INTERVAL
# =============================================================================


def parse_event_interval(system_index):
    """
    Parse:

        DFO_2049_From_20020903_to_20020906

    into:

        2002-09-03 00:00
        2002-09-06 23:59:59.999999

    ALL OUTPUT IS UTC-NAIVE.
    """

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

    start_string = match.group(1)
    end_string = match.group(2)

    try:

        start = pd.Timestamp(
            year=int(start_string[0:4]),
            month=int(start_string[4:6]),
            day=int(start_string[6:8]),
        )

        end_day = pd.Timestamp(
            year=int(end_string[0:4]),
            month=int(end_string[4:6]),
            day=int(end_string[6:8]),
        )

        end = end_day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

        return start, end

    except Exception:

        return pd.NaT, pd.NaT


# =============================================================================
# BUILD EVENT INTERVALS
# =============================================================================


def build_event_intervals(gdf):

    print()
    print("=" * 80)
    print("BUILDING GDF EVENT INTERVALS")
    print("=" * 80)

    intervals = []

    for _, row in gdf.iterrows():

        start, end = parse_event_interval(row["system_index"])

        if pd.isna(start) or pd.isna(end):
            continue

        intervals.append(
            {
                "event_start": start,
                "event_end": end,
                "gdf_id": row["id"],
                "system_index": str(row["system_index"]),
                "flood_fraction": row["flood_fraction"],
                "flooded_area_m2": row["flooded_area_m2"],
                "max_flood_duration": row["max_flood_duration"],
                "mean_clear_percentage": row["mean_clear_percentage"],
                "valid_area_m2": row["valid_area_m2"],
                "gdf_flood_occurred": int(row["flood_occurred"]),
                "event_time": row["event_time"],
            }
        )

    events = pd.DataFrame(intervals)

    if events.empty:

        raise RuntimeError("No valid GDF event intervals could be parsed.")

    # -------------------------------------------------------------------------
    # Force timestamp dtype one more time.
    # -------------------------------------------------------------------------

    for column in [
        "event_start",
        "event_end",
        "event_time",
    ]:

        if column in events.columns:

            events[column] = pd.Series(
                pd.array(
                    events[column],
                    dtype="datetime64[ns]",
                ),
                index=events.index,
            )

    events = events.sort_values("event_start").reset_index(drop=True)

    print()
    print(
        "Valid event intervals:",
        f"{len(events):,}",
    )

    print()
    print(
        "GDF period:",
        events["event_start"].min(),
        "→",
        events["event_end"].max(),
    )

    return events


# =============================================================================
# FILTER EVENTS TO ERA5 YEAR
# =============================================================================


def filter_events_to_year(
    events,
    era5,
):

    era5_start = era5["timestamp"].min()

    era5_end = era5["timestamp"].max()

    mask = (events["event_start"] <= era5_end) & (events["event_end"] >= era5_start)

    filtered = events.loc[mask].copy().reset_index(drop=True)

    print()
    print("=" * 80)
    print("MATCHING GDF EVENTS TO ERA5 PERIOD")
    print("=" * 80)

    print()
    print(
        "ERA5:",
        era5_start,
        "→",
        era5_end,
    )

    print()
    print(
        "Overlapping GDF events:",
        f"{len(filtered):,}",
    )

    if not filtered.empty:

        print()
        print(
            "Matching GDF period:",
            filtered["event_start"].min(),
            "→",
            filtered["event_end"].max(),
        )

    return filtered


# =============================================================================
# APPLY GDF LABELS
# =============================================================================


def apply_gdf_labels(
    era5,
    events,
):
    """
    High-performance interval matching.

    IMPORTANT:

    All actual matching is performed using INTEGER nanoseconds.

    Therefore there is NO possibility of:

        tz-aware vs tz-naive comparison

    during matching.

    ERA5:

        datetime64[ns]

    GDF:

        datetime64[ns]

    Matching representation:

        int64 nanoseconds since Unix epoch.

    For each event:

        start_ns <= ERA5_timestamp_ns <= end_ns

    This is extremely fast for large ERA5 datasets.
    """

    print()
    print("=" * 80)
    print("MATCHING GFD FLOOD EVENTS TO ERA5 TIMESTAMPS")
    print("=" * 80)

    # =========================================================================
    # COPY
    # =========================================================================

    result = era5.copy()

    events = events.copy()

    # =========================================================================
    # NORMALIZE AGAIN — DEFENSIVE
    # =========================================================================

    print()
    print("Normalizing ERA5 timestamps...")

    result["timestamp"] = normalize_timestamp_series(result["timestamp"])

    result = result[result["timestamp"].notna()].copy()

    print("Normalizing GDF timestamps...")

    for column in [
        "event_time",
        "event_start",
        "event_end",
    ]:

        if column in events.columns:

            events[column] = normalize_timestamp_series(events[column])

    events = events[events["event_start"].notna() & events["event_end"].notna()].copy()

    # =========================================================================
    # SORT
    # =========================================================================

    result = result.sort_values("timestamp").reset_index(drop=True)

    events = events.sort_values("event_start").reset_index(drop=True)

    # =========================================================================
    # ERA5 RANGE
    # =========================================================================

    era5_start = result["timestamp"].min()

    era5_end = result["timestamp"].max()

    print()
    print("ERA5 timestamp range:")

    print(f"  {era5_start}")

    print(f"  {era5_end}")

    print()
    print(f"ERA5 rows available: {len(result):,}")

    # =========================================================================
    # FILTER EVENTS
    # =========================================================================

    events = events.loc[
        (events["event_end"] >= era5_start) & (events["event_start"] <= era5_end)
    ].copy()

    events = events.sort_values("event_start").reset_index(drop=True)

    print()
    print(
        "GFD events overlapping ERA5:",
        f"{len(events):,}",
    )

    # =========================================================================
    # INITIALIZE OUTPUT COLUMNS
    # =========================================================================

    result["flood_occurred"] = np.int8(0)

    result["gdf_event_id"] = pd.Series(
        pd.array(
            [pd.NA] * len(result),
            dtype="Int64",
        ),
        index=result.index,
    )

    # -------------------------------------------------------------------------
    # CRITICAL:
    # datetime64[ns] column, NOT object and NOT timezone-aware.
    # -------------------------------------------------------------------------

    result["gdf_event_time"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns]",
    )

    result["gdf_system_index"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    result["gdf_flood_fraction"] = np.float64(0.0)

    result["gdf_flooded_area_m2"] = np.float64(0.0)

    result["gdf_max_flood_duration"] = np.float64(0.0)

    result["gdf_mean_clear_percentage"] = np.float64(np.nan)

    result["gdf_valid_area_m2"] = np.float64(0.0)

    # =========================================================================
    # NO EVENTS
    # =========================================================================

    if events.empty:

        print()
        print("No GDF events overlap the ERA5 period.")

        return result

    # =========================================================================
    # INTEGER TIMESTAMP ARRAY
    # =========================================================================
    #
    # Convert ERA5 timestamps to int64 nanoseconds.
    #
    # This completely eliminates timezone comparison problems.
    #
    # ==========================================================================

    era5_ns = result["timestamp"].astype("int64").to_numpy()

    # =========================================================================
    # MATCH EVENTS
    # =========================================================================

    matched_rows = 0

    for event_index, event in events.iterrows():

        event_start = event["event_start"]

        event_end = event["event_end"]

        # ---------------------------------------------------------------------
        # Clamp to ERA5 period
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
        # Convert BOTH endpoints to int64 nanoseconds.
        #
        # No pandas/NumPy timezone interaction anymore.
        # ---------------------------------------------------------------------

        start_ns = int(match_start.value)

        end_ns = int(match_end.value)

        # ---------------------------------------------------------------------
        # FAST RANGE LOOKUP
        #
        # ERA5 is sorted.
        #
        # searchsorted receives only int64 arrays.
        # ---------------------------------------------------------------------

        left = np.searchsorted(
            era5_ns,
            start_ns,
            side="left",
        )

        right = np.searchsorted(
            era5_ns,
            end_ns,
            side="right",
        )

        # ---------------------------------------------------------------------
        # No matching ERA5 timestamps.
        # ---------------------------------------------------------------------

        if right <= left:

            print(
                f"  Event {event_index + 1:03d}: "
                f"{event_start} → {event_end} "
                f"→ 0 ERA5 rows"
            )

            continue

        # ---------------------------------------------------------------------
        # Integer positional range.
        #
        # No boolean datetime comparison required.
        # ---------------------------------------------------------------------

        positions = np.arange(
            left,
            right,
            dtype=np.int64,
        )

        count = len(positions)

        # ---------------------------------------------------------------------
        # Flood flag
        # ---------------------------------------------------------------------

        result.iloc[positions, result.columns.get_loc("flood_occurred")] = 1

        # ---------------------------------------------------------------------
        # Event time
        # ---------------------------------------------------------------------

        event_time = event.get(
            "event_time",
            pd.NaT,
        )

        if pd.notna(event_time):

            event_time = normalize_timestamp_utc_naive(event_time)

            result.iloc[positions, result.columns.get_loc("gdf_event_time")] = (
                event_time
            )

        # ---------------------------------------------------------------------
        # Event ID
        # ---------------------------------------------------------------------

        event_id = event.get(
            "gdf_id",
            np.nan,
        )

        if pd.notna(event_id):

            try:

                event_id = int(float(event_id))

                result.iloc[positions, result.columns.get_loc("gdf_event_id")] = (
                    event_id
                )

            except (
                ValueError,
                TypeError,
            ):

                pass

        # ---------------------------------------------------------------------
        # SYSTEM INDEX
        # ---------------------------------------------------------------------

        system_index = event.get(
            "system_index",
            None,
        )

        if pd.notna(system_index):

            result.iloc[positions, result.columns.get_loc("gdf_system_index")] = str(
                system_index
            )

        # ---------------------------------------------------------------------
        # Helper for numeric values
        # ---------------------------------------------------------------------

        def numeric_value(
            column,
            default=None,
        ):

            value = event.get(
                column,
                default,
            )

            value = pd.to_numeric(
                value,
                errors="coerce",
            )

            return value

        # ---------------------------------------------------------------------
        # FLOOD FRACTION
        # ---------------------------------------------------------------------

        value = numeric_value("flood_fraction")

        if pd.notna(value):

            result.iloc[positions, result.columns.get_loc("gdf_flood_fraction")] = (
                float(value)
            )

        # ---------------------------------------------------------------------
        # FLOODED AREA
        # ---------------------------------------------------------------------

        value = numeric_value("flooded_area_m2")

        if pd.notna(value):

            result.iloc[positions, result.columns.get_loc("gdf_flooded_area_m2")] = (
                float(value)
            )

        # ---------------------------------------------------------------------
        # MAX FLOOD DURATION
        # ---------------------------------------------------------------------

        value = numeric_value("max_flood_duration")

        if pd.notna(value):

            result.iloc[positions, result.columns.get_loc("gdf_max_flood_duration")] = (
                float(value)
            )

        # ---------------------------------------------------------------------
        # CLEAR PERCENTAGE
        # ---------------------------------------------------------------------

        value = numeric_value("mean_clear_percentage")

        if pd.notna(value):

            result.iloc[
                positions, result.columns.get_loc("gdf_mean_clear_percentage")
            ] = float(value)

        # ---------------------------------------------------------------------
        # VALID AREA
        # ---------------------------------------------------------------------

        value = numeric_value("valid_area_m2")

        if pd.notna(value):

            result.iloc[positions, result.columns.get_loc("gdf_valid_area_m2")] = float(
                value
            )

        # ---------------------------------------------------------------------
        # STATISTICS
        # ---------------------------------------------------------------------

        matched_rows += count

        print(
            f"  Event {event_index + 1:03d}: "
            f"{event_start} → {event_end} "
            f"→ {count:,} ERA5 rows"
        )

    # =========================================================================
    # FINAL STATISTICS
    # =========================================================================

    flood_rows = int(result["flood_occurred"].sum())

    print()
    print("=" * 80)
    print("GFD ↔ ERA5 MATCH COMPLETE")
    print("=" * 80)

    print()
    print(
        "ERA5 rows:",
        f"{len(result):,}",
    )

    print(
        "GFD events:",
        f"{len(events):,}",
    )

    print(
        "Matched ERA5 rows:",
        f"{flood_rows:,}",
    )

    print(
        "Unmatched ERA5 rows:",
        f"{len(result) - flood_rows:,}",
    )

    print(
        "Total event-row matches:",
        f"{matched_rows:,}",
    )

    if len(result) > 0:

        print("Flood-labelled percentage:", f"{flood_rows / len(result) * 100:.4f}%")

    return result


# =============================================================================
# FINAL COLUMN ORDER
# =============================================================================


def arrange_columns(df):

    era5_columns = [
        "timestamp",
        "city",
    ] + FEATURES

    gdf_columns = [
        "flood_occurred",
        "gdf_event_id",
        "gdf_system_index",
        "gdf_event_time",
        "gdf_flood_fraction",
        "gdf_flooded_area_m2",
        "gdf_max_flood_duration",
        "gdf_mean_clear_percentage",
        "gdf_valid_area_m2",
    ]

    columns = era5_columns + gdf_columns

    columns = [column for column in columns if column in df.columns]

    return df[columns].copy()


# =============================================================================
# VALIDATION
# =============================================================================


def validate_result(df):

    print()
    print("=" * 80)
    print("FINAL VALIDATION")
    print("=" * 80)

    print()
    print("Rows:")
    print(f"{len(df):,}")

    print()
    print("Columns:")
    print(len(df.columns))

    print()
    print(
        "40 SOALM features:",
        sum(feature in df.columns for feature in FEATURES),
        "/ 40",
    )

    # -------------------------------------------------------------------------
    # Flood labels
    # -------------------------------------------------------------------------

    print()
    print("Flood labels:")

    print(df["flood_occurred"].value_counts().sort_index().to_string())

    # -------------------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------------------

    print()
    print("Timestamp:")

    print(
        "First:",
        df["timestamp"].min(),
    )

    print(
        "Last: ",
        df["timestamp"].max(),
    )

    print(
        "dtype:",
        df["timestamp"].dtype,
    )

    # -------------------------------------------------------------------------
    # Check timezone
    # -------------------------------------------------------------------------

    if hasattr(
        df["timestamp"].dtype,
        "tz",
    ):

        if df["timestamp"].dtype.tz is not None:

            raise RuntimeError(
                "Timestamp is still timezone-aware. " "This should never happen."
            )

    # -------------------------------------------------------------------------
    # Required GDF columns
    # -------------------------------------------------------------------------

    required_gdf_output = [
        "flood_occurred",
        "gdf_event_id",
        "gdf_system_index",
        "gdf_event_time",
        "gdf_flood_fraction",
        "gdf_flooded_area_m2",
        "gdf_max_flood_duration",
        "gdf_mean_clear_percentage",
        "gdf_valid_area_m2",
    ]

    missing_output = [
        column for column in required_gdf_output if column not in df.columns
    ]

    if missing_output:

        raise RuntimeError("Missing GDF output columns:\n" + "\n".join(missing_output))

    print()
    print("GDF output columns:")
    print("✓ All required GDF columns present")

    # -------------------------------------------------------------------------
    # Flood examples
    # -------------------------------------------------------------------------

    flood_df = df[df["flood_occurred"] == 1]

    if not flood_df.empty:

        print()
        print("First flood-labelled rows:")

        print(
            flood_df[
                [
                    "timestamp",
                    "flood_occurred",
                    "gdf_event_id",
                    "gdf_system_index",
                    "gdf_event_time",
                    "gdf_flood_fraction",
                    "gdf_flooded_area_m2",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    else:

        print()
        print("⚠ No flood rows were found " "in this ERA5 year.")


# =============================================================================
# SAVE
# =============================================================================


def save_dataset(df):

    PREPARED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_FILE.exists():

        if not OVERWRITE:

            print()
            print("Output already exists and " "OVERWRITE=False:")

            print(OUTPUT_FILE)

            return

        print()
        print("Overwriting existing output:")

        print(OUTPUT_FILE)

        OUTPUT_FILE.unlink()

    print()
    print("=" * 80)
    print("SAVING COMBINED DATASET")
    print("=" * 80)

    df.to_parquet(
        OUTPUT_FILE,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    size_gb = OUTPUT_FILE.stat().st_size / (1024**3)

    print()
    print("✓ Saved successfully")

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
    print("Size:")
    print(f"{size_gb:.3f} GB")


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
        # 3. BUILD EVENT INTERVALS
        # =====================================================================

        events = build_event_intervals(gdf)

        # =====================================================================
        # 4. FILTER EVENTS TO YEAR
        # =====================================================================

        events = filter_events_to_year(
            events,
            era5,
        )

        # =====================================================================
        # 5. MATCH
        # =====================================================================

        combined = apply_gdf_labels(
            era5,
            events,
        )

        # =====================================================================
        # 6. ARRANGE
        # =====================================================================

        combined = arrange_columns(combined)

        # =====================================================================
        # 7. VALIDATE
        # =====================================================================

        validate_result(combined)

        # =====================================================================
        # 8. SAVE
        # =====================================================================

        save_dataset(combined)

        # =====================================================================
        # COMPLETE
        # =====================================================================

        print()
        print("=" * 80)
        print("COMPLETE")
        print("=" * 80)

        print()
        print("Combined Surat dataset:")

        print(OUTPUT_FILE)

        print()
        print(
            "Next step: inspect the flood-labelled rows "
            "before creating the final training targets."
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


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()
