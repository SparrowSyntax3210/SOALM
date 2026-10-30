"""
===============================================================================
SOALM — GLOBAL FLOOD DATABASE DOWNLOADER
SURAT TEST DATASET

Downloads GFD flood-event metadata + flood footprint statistics.

Output:
    data/flood_training/surat/raw/gfd/

Files:
    gfd_surat_events.csv
    gfd_surat_flood_stats.csv

IMPORTANT:
    This overwrites the previous Surat GFD output.
===============================================================================
"""

from pathlib import Path
import ee
import pandas as pd
import time

# =============================================================================
# CONFIG
# =============================================================================

LATITUDE = 21.1702
LONGITUDE = 72.8311

RADIUS_KM = 100

CITY = "surat"

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "data" / "flood_training" / CITY / "raw" / "gfd"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


EVENT_FILE = OUTPUT_DIR / "gfd_surat_events.csv"
STATS_FILE = OUTPUT_DIR / "gfd_surat_flood_stats.csv"


# =============================================================================
# EARTH ENGINE
# =============================================================================

print("=" * 75)
print("SOALM — GLOBAL FLOOD DATABASE")
print("=" * 75)

print()
print("Initializing Google Earth Engine...")

try:

    ee.Initialize(project="soalm-1207")

    print("✓ Earth Engine initialized")

except Exception as e:

    print()
    print("Earth Engine initialization failed.")
    print()
    print(e)

    raise


# =============================================================================
# SEARCH REGION
# =============================================================================

# Approximate conversion.
# For the first Surat experiment this is sufficient.

lat_delta = RADIUS_KM / 111.0

lon_delta = RADIUS_KM / (
    111.0 * max(0.01, __import__("math").cos(__import__("math").radians(LATITUDE)))
)


region = ee.Geometry.Rectangle(
    [
        LONGITUDE - lon_delta,
        LATITUDE - lat_delta,
        LONGITUDE + lon_delta,
        LATITUDE + lat_delta,
    ]
)


print()
print("Location:")
print("  Latitude :", LATITUDE)
print("  Longitude:", LONGITUDE)

print()
print("Search radius:")
print(" ", RADIUS_KM, "km")


# =============================================================================
# LOAD GFD
# =============================================================================

print()
print("Loading Global Flood Database...")

GFD = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")

print("✓ GFD collection loaded")


# =============================================================================
# SEARCH EVENTS
# =============================================================================

print()
print("Searching flood events around Surat...")

events = GFD.filterBounds(region)

event_count = events.size().getInfo()

print()
print("Events found:")
print(" ", event_count)


if event_count == 0:

    print()
    print("No GFD events found.")

    raise SystemExit(0)


# =============================================================================
# EVENT METADATA
# =============================================================================

print()
print("Downloading event metadata...")


def extract_metadata(image):

    return ee.Feature(
        None,
        {
            "system_index": image.get("system:index"),
            "system_time_start": image.get("system:time_start"),
            "id": image.get("id"),
            "dfo_centroid_x": image.get("dfo_centroid_x"),
            "dfo_centroid_y": image.get("dfo_centroid_y"),
            "dfo_country": image.get("dfo_country"),
            "dfo_main_cause": image.get("dfo_main_cause"),
            "dfo_other_cause": image.get("dfo_other_cause"),
            "dfo_dead": image.get("dfo_dead"),
            "dfo_displaced": image.get("dfo_displaced"),
            "dfo_began": image.get("dfo_began"),
            "dfo_ended": image.get("dfo_ended"),
        },
    )


metadata_fc = ee.FeatureCollection(events.map(extract_metadata))

metadata = metadata_fc.getInfo()

rows = []

for feature in metadata["features"]:

    rows.append(feature.get("properties", {}))


events_df = pd.DataFrame(rows)

events_df.to_csv(EVENT_FILE, index=False)


print()
print("✓ Event metadata saved:")
print(EVENT_FILE)

print()
print("Rows:")
print(len(events_df))


# =============================================================================
# FLOOD FOOTPRINT STATISTICS
# =============================================================================

print()
print("=" * 75)
print("CALCULATING FLOOD FOOTPRINT")
print("=" * 75)

print()
print("Using bands:")
print("  flooded")
print("  duration")
print("  clear_views")
print("  clear_perc")
print("  jrc_perm_water")


# =============================================================================
# STATISTICS FUNCTION
# =============================================================================


def calculate_event_stats(image):

    flooded = image.select("flooded")

    duration = image.select("duration")

    clear_views = image.select("clear_views")

    clear_perc = image.select("clear_perc")

    permanent_water = image.select("jrc_perm_water")

    # -------------------------------------------------------------------------
    # Valid observation pixels
    #
    # clear_views > 0 means the location had an observation.
    # -------------------------------------------------------------------------

    valid = clear_views.gt(0)

    # -------------------------------------------------------------------------
    # Flood pixels
    #
    # flooded == 1 means flood detected.
    # Exclude permanent water.
    # -------------------------------------------------------------------------

    flood_pixels = flooded.eq(1).And(permanent_water.eq(0)).And(valid)

    # -------------------------------------------------------------------------
    # Area calculations
    # -------------------------------------------------------------------------

    pixel_area = ee.Image.pixelArea()

    flooded_area = (
        pixel_area.updateMask(flood_pixels)
        .reduceRegion(
            reducer=ee.Reducer.sum(), geometry=region, scale=250, maxPixels=1e9
        )
        .get("area")
    )

    valid_area = (
        pixel_area.updateMask(valid)
        .reduceRegion(
            reducer=ee.Reducer.sum(), geometry=region, scale=250, maxPixels=1e9
        )
        .get("area")
    )

    # -------------------------------------------------------------------------
    # Flood fraction
    # -------------------------------------------------------------------------

    flood_fraction = ee.Number(flooded_area).divide(ee.Number(valid_area))

    # -------------------------------------------------------------------------
    # Maximum flood duration
    # -------------------------------------------------------------------------

    max_duration = (
        duration.updateMask(flood_pixels)
        .reduceRegion(
            reducer=ee.Reducer.max(), geometry=region, scale=250, maxPixels=1e9
        )
        .get("duration")
    )

    # -------------------------------------------------------------------------
    # Mean clear percentage
    # -------------------------------------------------------------------------

    mean_clear = (
        clear_perc.updateMask(valid)
        .reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=250, maxPixels=1e9
        )
        .get("clear_perc")
    )

    return ee.Feature(
        None,
        {
            "system_index": image.get("system:index"),
            "id": image.get("id"),
            "event_time": image.get("system:time_start"),
            "flood_fraction": flood_fraction,
            "flooded_area_m2": flooded_area,
            "valid_area_m2": valid_area,
            "max_flood_duration": max_duration,
            "mean_clear_percentage": mean_clear,
        },
    )


# =============================================================================
# RUN
# =============================================================================

print()
print("Processing", event_count, "events...")

stats_fc = ee.FeatureCollection(events.map(calculate_event_stats))


print()
print("Requesting statistics from Earth Engine...")

stats_info = stats_fc.getInfo()


stats_rows = []

for feature in stats_info["features"]:

    stats_rows.append(feature.get("properties", {}))


stats_df = pd.DataFrame(stats_rows)


# =============================================================================
# CLEAN NUMBERS
# =============================================================================

numeric_columns = [
    "flood_fraction",
    "flooded_area_m2",
    "valid_area_m2",
    "max_flood_duration",
    "mean_clear_percentage",
]


for column in numeric_columns:

    if column in stats_df.columns:

        stats_df[column] = pd.to_numeric(stats_df[column], errors="coerce")


# =============================================================================
# FLOOD LABEL
# =============================================================================

stats_df["flood_occurred"] = stats_df["flood_fraction"].fillna(0).gt(0).astype(int)


# =============================================================================
# SAVE
# =============================================================================

stats_df.to_csv(STATS_FILE, index=False)


print()
print("=" * 75)
print("✓ FLOOD STATISTICS SAVED")
print("=" * 75)

print()
print("File:")
print(STATS_FILE)

print()
print("Rows:")
print(len(stats_df))

print()
print("Flood events with detected inundation:", int(stats_df["flood_occurred"].sum()))


# =============================================================================
# PREVIEW
# =============================================================================

print()
print("=" * 75)
print("PREVIEW")
print("=" * 75)

preview_columns = [
    "id",
    "flood_fraction",
    "flooded_area_m2",
    "valid_area_m2",
    "max_flood_duration",
    "mean_clear_percentage",
    "flood_occurred",
]


available = [c for c in preview_columns if c in stats_df.columns]


print(stats_df[available].head(10).to_string(index=False))


# =============================================================================
# COMPLETE
# =============================================================================

print()
print("=" * 75)
print("GFD DOWNLOAD COMPLETE")
print("=" * 75)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Next step:")
print("Merge these GFD event observations with " "the 6-hour Surat ERA5 data.")

print()
print("Do NOT train the flood model yet.")
print()
