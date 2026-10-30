"""
===============================================================================
SOALM — SENTINEL-1 SURAT DOWNLOADER
===============================================================================

PURPOSE
-------
Download/search Sentinel-1 SAR observations for Surat around GDF flood events.

IMPORTANT
---------
Sentinel-1 became operational in 2014.

Therefore:

    GDF 2000 → no Sentinel-1
    GDF 2002 → no Sentinel-1
    ...
    GDF 2014+ → Sentinel-1 may be available

The GDF dataset determines the flood-event windows.

ERA5 is NOT used to create Sentinel-1 observations. ERA5 and Sentinel-1
remain independent data sources and can later be merged using timestamps.

OUTPUT
------
Raw Sentinel-1 metadata:

    data/sentinel1/surat/raw/

Parquet index:

    data/sentinel1/surat/sentinel1_index.parquet

The index contains:

    event_id
    gdf_system_index
    gdf_event_start
    gdf_event_end
    sentinel_datetime
    product_id
    collection
    platform
    orbit_direction
    relative_orbit
    polarization
    product_type
    cloud_cover
    geometry
    bbox
    download_url

===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from datetime import timedelta
import getpass
import json
import sys
import traceback

import requests
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GDF_FILE = (
    PROJECT_ROOT / "data" / "flood_training" / "surat" / "raw" / "gfd" / "surat.csv"
)

ERA5_ROOT = PROJECT_ROOT / "data" / "era5" / "features" / "surat"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "sentinel1" / "surat"

RAW_ROOT = OUTPUT_ROOT / "raw"

INDEX_FILE = OUTPUT_ROOT / "sentinel1_index.parquet"


# =============================================================================
# SURAT SEARCH AREA
# =============================================================================

MIN_LON = 72.65
MIN_LAT = 21.05
MAX_LON = 72.95
MAX_LAT = 21.30


# =============================================================================
# SENTINEL-1
# =============================================================================

# Correct Copernicus Data Space STAC collection.
SENTINEL_COLLECTION = "sentinel-1-grd"

# Sentinel-1 started acquiring data in 2014.
SENTINEL_START = pd.Timestamp(
    "2014-01-01",
    tz="UTC",
)

# Search around each GDF event.
SEARCH_BEFORE_DAYS = 5
SEARCH_AFTER_DAYS = 5

# Limit number of returned products per event.
MAX_PRODUCTS_PER_EVENT = 50


# =============================================================================
# COPERNICUS DATA SPACE
# =============================================================================

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)

STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"

DOWNLOAD_BASE = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"


# =============================================================================
# HTTP
# =============================================================================

USER_AGENT = "SOALM-Sentinel1-Downloader/1.0 " "(research flood intelligence project)"

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
)


# =============================================================================
# HEADER
# =============================================================================


def print_header():

    print()
    print("=" * 80)
    print("SOALM — SENTINEL-1 SURAT DOWNLOADER")
    print("=" * 80)

    print()
    print("GDF:")
    print(GDF_FILE)

    print()
    print("ERA5:")
    print(ERA5_ROOT)

    print()
    print("Sentinel-1 raw output:")
    print(RAW_ROOT)

    print()
    print("Parquet index:")
    print(INDEX_FILE)

    print()
    print("Search area:")
    print(f"{MIN_LON}, {MIN_LAT} → " f"{MAX_LON}, {MAX_LAT}")

    print()
    print("Sentinel-1 collection:")
    print(SENTINEL_COLLECTION)

    print()
    print("Sentinel-1 availability:")
    print(f"{SENTINEL_START}")

    print()
    print(
        "Search window:",
        f"-{SEARCH_BEFORE_DAYS} days → " f"+{SEARCH_AFTER_DAYS} days",
    )

    print()
    print("=" * 80)


# =============================================================================
# LOAD GDF
# =============================================================================


def load_gdf():

    print()
    print("=" * 80)
    print("LOADING GDF")
    print("=" * 80)

    if not GDF_FILE.exists():

        raise FileNotFoundError(f"\nGDF file not found:\n{GDF_FILE}\n")

    gdf = pd.read_csv(GDF_FILE)

    print()
    print("GDF rows:")
    print(f"{len(gdf):,}")

    required = [
        "event_time",
        "system_index",
    ]

    missing = [c for c in required if c not in gdf.columns]

    if missing:

        raise RuntimeError("Missing GDF columns:\n" + "\n".join(missing))

    # -------------------------------------------------------------------------
    # event_time
    # -------------------------------------------------------------------------

    raw_time = gdf["event_time"]

    numeric = pd.to_numeric(
        raw_time,
        errors="coerce",
    )

    gdf["event_time"] = pd.to_datetime(
        numeric,
        unit="ms",
        errors="coerce",
        utc=True,
    )

    gdf = gdf.dropna(subset=["event_time"]).copy()

    print()
    print("Valid GDF timestamps:")
    print(f"{len(gdf):,}")

    return gdf


# =============================================================================
# PARSE GDF EVENT
# =============================================================================


def parse_event_interval(system_index):

    if pd.isna(system_index):

        return pd.NaT, pd.NaT

    text = str(system_index)

    # Example:
    #
    # DFO_2680_From_20050628_to_20050715

    import re

    pattern = r"From_(\d{8})_to_(\d{8})"

    match = re.search(
        pattern,
        text,
    )

    if not match:

        return pd.NaT, pd.NaT

    start_text = match.group(1)
    end_text = match.group(2)

    try:

        start = pd.Timestamp(
            start_text,
            tz="UTC",
        )

        end = (
            pd.Timestamp(
                end_text,
                tz="UTC",
            )
            + pd.Timedelta(days=1)
            - pd.Timedelta(seconds=1)
        )

        return start, end

    except Exception:

        return pd.NaT, pd.NaT


# =============================================================================
# BUILD EVENTS
# =============================================================================


def build_events(gdf):

    print()
    print("=" * 80)
    print("BUILDING GDF FLOOD EVENTS")
    print("=" * 80)

    events = []

    for index, row in gdf.iterrows():

        start, end = parse_event_interval(row["system_index"])

        if pd.isna(start) or pd.isna(end):

            continue

        events.append(
            {
                "event_number": len(events) + 1,
                "gdf_row": index,
                "gdf_id": row.get("id"),
                "gdf_system_index": row["system_index"],
                "gdf_event_time": row["event_time"],
                "gdf_event_start": start,
                "gdf_event_end": end,
            }
        )

    events = pd.DataFrame(events)

    if events.empty:

        raise RuntimeError("No valid GDF flood events found.")

    events = (
        events.drop_duplicates(
            subset=["gdf_system_index"],
        )
        .sort_values("gdf_event_start")
        .reset_index(drop=True)
    )

    print()
    print("Unique flood events:")
    print(f"{len(events):,}")

    print()
    print("GDF period:")
    print(
        events["gdf_event_start"].min(),
        "→",
        events["gdf_event_end"].max(),
    )

    return events


# =============================================================================
# ERA5 RANGE
# =============================================================================


def load_era5_range():

    print()
    print("=" * 80)
    print("CHECKING ERA5")
    print("=" * 80)

    if not ERA5_ROOT.exists():

        print()
        print("⚠ ERA5 Surat directory does not exist.")

        return None, None

    files = list(ERA5_ROOT.glob("*.parquet"))

    if not files:

        print()
        print("⚠ No ERA5 parquet files found.")

        return None, None

    all_start = None
    all_end = None

    for file in files:

        try:

            df = pd.read_parquet(
                file,
                columns=["timestamp"],
            )

            if df.empty:
                continue

            timestamps = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
                utc=True,
            ).dropna()

            if timestamps.empty:
                continue

            start = timestamps.min()
            end = timestamps.max()

            if all_start is None or start < all_start:
                all_start = start

            if all_end is None or end > all_end:
                all_end = end

        except Exception as exc:

            print(f"⚠ Could not inspect {file.name}: {exc}")

    print()
    print("ERA5 period:")

    if all_start is not None:

        print(f"{all_start} → {all_end}")

    else:

        print("Unavailable")

    return all_start, all_end


# =============================================================================
# AUTHENTICATION
# =============================================================================


def authenticate():

    print()
    print("=" * 80)
    print("COPERNICUS DATA SPACE LOGIN")
    print("=" * 80)

    print()
    print("Create/login to your Copernicus Data Space account:")

    print("https://dataspace.copernicus.eu/")

    print()

    username = input("Copernicus username/email: ").strip()

    password = getpass.getpass("Copernicus password: ")

    payload = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }

    print()
    print("Authenticating with Copernicus Data Space...")

    response = SESSION.post(
        TOKEN_URL,
        data=payload,
        timeout=30,
    )

    if response.status_code != 200:

        raise RuntimeError(
            "\nCopernicus authentication failed.\n"
            f"HTTP: {response.status_code}\n"
            f"{response.text}\n"
        )

    token = response.json().get("access_token")

    if not token:

        raise RuntimeError("Authentication succeeded but no access token was returned.")

    SESSION.headers.update(
        {
            "Authorization": f"Bearer {token}",
        }
    )

    print()
    print("✓ Authentication successful")

    return token


# =============================================================================
# STAC SEARCH
# =============================================================================


def search_sentinel1(
    start,
    end,
):

    # -------------------------------------------------------------------------
    # GeoJSON polygon
    # -------------------------------------------------------------------------

    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [MIN_LON, MIN_LAT],
                [MAX_LON, MIN_LAT],
                [MAX_LON, MAX_LAT],
                [MIN_LON, MAX_LAT],
                [MIN_LON, MIN_LAT],
            ]
        ],
    }

    payload = {
        "collections": [SENTINEL_COLLECTION],
        "datetime": (f"{start.isoformat()}/" f"{end.isoformat()}"),
        "intersects": geometry,
        "limit": MAX_PRODUCTS_PER_EVENT,
    }

    response = SESSION.post(
        STAC_URL,
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:

        raise RuntimeError(
            "\nSentinel-1 STAC search failed.\n"
            f"HTTP: {response.status_code}\n"
            f"{response.text}\n"
        )

    data = response.json()

    return data.get("features", [])


# =============================================================================
# PRODUCT METADATA
# =============================================================================


def extract_product(
    feature,
    event,
):

    properties = feature.get(
        "properties",
        {},
    )

    product_id = (
        feature.get("id")
        or properties.get("title")
        or properties.get("productIdentifier")
    )

    # -------------------------------------------------------------------------
    # Sentinel platform
    # -------------------------------------------------------------------------

    platform = properties.get("platform")

    if not platform:

        platform = properties.get("sat:platform")

    # -------------------------------------------------------------------------
    # Orbit
    # -------------------------------------------------------------------------

    orbit_direction = properties.get("sat:orbit_state") or properties.get(
        "orbitDirection"
    )

    relative_orbit = properties.get("sat:relative_orbit") or properties.get(
        "relativeOrbitNumber"
    )

    # -------------------------------------------------------------------------
    # Polarization
    # -------------------------------------------------------------------------

    polarization = properties.get("sar:polarizations") or properties.get(
        "polarizations"
    )

    if isinstance(
        polarization,
        list,
    ):

        polarization = ",".join(map(str, polarization))

    # -------------------------------------------------------------------------
    # Product type
    # -------------------------------------------------------------------------

    product_type = properties.get("productType") or properties.get("s1:product_type")

    # -------------------------------------------------------------------------
    # Cloud cover
    #
    # Sentinel-1 SAR is essentially cloud-independent,
    # but retain this if the catalogue provides it.
    # -------------------------------------------------------------------------

    cloud_cover = properties.get("eo:cloud_cover")

    # -------------------------------------------------------------------------
    # Datetime
    # -------------------------------------------------------------------------

    sentinel_datetime = feature.get("datetime") or properties.get("datetime")

    sentinel_datetime = pd.to_datetime(
        sentinel_datetime,
        errors="coerce",
        utc=True,
    )

    # -------------------------------------------------------------------------
    # Download URL
    # -------------------------------------------------------------------------

    download_url = None

    if product_id:

        download_url = f"{DOWNLOAD_BASE}(" f"'{product_id}'" f")/$value"

    return {
        "event_number": event["event_number"],
        "gdf_row": event["gdf_row"],
        "gdf_id": event["gdf_id"],
        "gdf_system_index": event["gdf_system_index"],
        "gdf_event_time": event["gdf_event_time"],
        "gdf_event_start": event["gdf_event_start"],
        "gdf_event_end": event["gdf_event_end"],
        "sentinel_datetime": sentinel_datetime,
        "product_id": product_id,
        "collection": SENTINEL_COLLECTION,
        "platform": platform,
        "orbit_direction": orbit_direction,
        "relative_orbit": relative_orbit,
        "polarization": polarization,
        "product_type": product_type,
        "cloud_cover": cloud_cover,
        "geometry": json.dumps(feature.get("geometry")),
        "bbox": json.dumps(feature.get("bbox")),
        "download_url": download_url,
        "stac_properties": json.dumps(properties),
    }


# =============================================================================
# DOWNLOAD PRODUCT
# =============================================================================


def download_product(
    product,
):

    product_id = product["product_id"]

    if not product_id:

        print("⚠ Product has no ID. Skipping.")

        return None

    RAW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Sentinel products can be very large.
    # -------------------------------------------------------------------------

    safe_name = str(product_id).replace("/", "_").replace("\\", "_").replace(":", "_")

    output_file = RAW_ROOT / f"{safe_name}.zip"

    if output_file.exists():

        print(
            "  ✓ Already downloaded:",
            output_file.name,
        )

        return output_file

    url = product["download_url"]

    if not url:

        return None

    print()
    print(
        "  Downloading:",
        product_id,
    )

    with SESSION.get(
        url,
        stream=True,
        timeout=120,
    ) as response:

        if response.status_code != 200:

            print(
                "  ⚠ Download failed:",
                response.status_code,
            )

            return None

        with open(
            output_file,
            "wb",
        ) as file:

            for chunk in response.iter_content(chunk_size=1024 * 1024):

                if chunk:

                    file.write(chunk)

    print(
        "  ✓ Saved:",
        output_file,
    )

    return output_file


# =============================================================================
# MAIN SEARCH
# =============================================================================


def run_search(
    events,
):

    print()
    print("=" * 80)
    print("SEARCHING SENTINEL-1")
    print("=" * 80)

    results = []

    for i, event in events.iterrows():

        event_start = event["gdf_event_start"]

        event_end = event["gdf_event_end"]

        print()
        print(f"Event {i + 1:03d}/{len(events):03d}")

        print(
            "GDF:",
            event["gdf_system_index"],
        )

        # ---------------------------------------------------------------------
        # Sentinel-1 did not exist before 2014.
        # ---------------------------------------------------------------------

        if event_end < SENTINEL_START:

            print("  → Sentinel-1 unavailable " "for this event.")

            continue

        # ---------------------------------------------------------------------
        # Clamp beginning to Sentinel-1 availability.
        # ---------------------------------------------------------------------

        search_start = max(
            event_start - pd.Timedelta(days=SEARCH_BEFORE_DAYS),
            SENTINEL_START,
        )

        search_end = event_end + pd.Timedelta(days=SEARCH_AFTER_DAYS)

        print(
            "Search:",
            search_start,
            "→",
            search_end,
        )

        try:

            features = search_sentinel1(
                search_start,
                search_end,
            )

        except Exception as exc:

            print("⚠ Search failed:")

            print(exc)

            continue

        print(
            "Products found:",
            len(features),
        )

        for feature in features:

            product = extract_product(
                feature,
                event,
            )

            results.append(product)

            print(
                "  ✓",
                product["sentinel_datetime"],
                "|",
                product["platform"],
                "|",
                product["product_id"],
            )

    return pd.DataFrame(results)


# =============================================================================
# SAVE INDEX
# =============================================================================


def save_index(
    df,
):

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if df.empty:

        print()
        print("⚠ No Sentinel-1 products found.")

        # Still save an empty parquet with known columns.
        columns = [
            "event_number",
            "gdf_row",
            "gdf_id",
            "gdf_system_index",
            "gdf_event_time",
            "gdf_event_start",
            "gdf_event_end",
            "sentinel_datetime",
            "product_id",
            "collection",
            "platform",
            "orbit_direction",
            "relative_orbit",
            "polarization",
            "product_type",
            "cloud_cover",
            "geometry",
            "bbox",
            "download_url",
            "stac_properties",
        ]

        df = pd.DataFrame(columns=columns)

    # -------------------------------------------------------------------------
    # Normalize timestamps before Parquet.
    # -------------------------------------------------------------------------

    timestamp_columns = [
        "gdf_event_time",
        "gdf_event_start",
        "gdf_event_end",
        "sentinel_datetime",
    ]

    for column in timestamp_columns:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
                utc=True,
            )

    # -------------------------------------------------------------------------
    # Remove duplicates.
    # -------------------------------------------------------------------------

    if "product_id" in df.columns:

        df = df.drop_duplicates(
            subset=[
                "gdf_system_index",
                "product_id",
            ]
        )

    # -------------------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------------------

    df.to_parquet(
        INDEX_FILE,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    print()
    print("=" * 80)
    print("SAVING SENTINEL-1 INDEX")
    print("=" * 80)

    print()
    print("File:")
    print(INDEX_FILE)

    print()
    print("Rows:")
    print(f"{len(df):,}")

    print()
    print("Columns:")
    print(len(df.columns))

    if not df.empty:

        print()
        print("Sentinel-1 period:")

        print(
            df["sentinel_datetime"].min(),
            "→",
            df["sentinel_datetime"].max(),
        )

        print()
        print("Unique products:")

        print(df["product_id"].nunique())

    print()
    print("✓ Parquet index saved")

    return df


# =============================================================================
# SUMMARY
# =============================================================================


def print_summary(
    events,
    products,
):

    print()
    print("=" * 80)
    print("SENTINEL-1 DOWNLOAD SUMMARY")
    print("=" * 80)

    total_events = len(events)

    sentinel_events = (events["gdf_event_end"] >= SENTINEL_START).sum()

    print()
    print(
        "Total GDF events:",
        f"{total_events:,}",
    )

    print(
        "Events after Sentinel-1 start:",
        f"{sentinel_events:,}",
    )

    print(
        "Events before Sentinel-1:",
        f"{total_events - sentinel_events:,}",
    )

    print()
    print(
        "Sentinel-1 products found:",
        f"{len(products):,}",
    )

    if not products.empty:

        print()
        print(
            "Unique Sentinel-1 products:",
            products["product_id"].nunique(),
        )

        print()
        print("Products per GDF event:")

        counts = products.groupby("gdf_system_index").size()

        print(counts.to_string())

    print()
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================


def main():

    print_header()

    try:

        # ---------------------------------------------------------------------
        # 1. Load GDF
        # ---------------------------------------------------------------------

        gdf = load_gdf()

        # ---------------------------------------------------------------------
        # 2. Build events
        # ---------------------------------------------------------------------

        events = build_events(gdf)

        # ---------------------------------------------------------------------
        # 3. Inspect ERA5
        # ---------------------------------------------------------------------

        load_era5_range()

        # ---------------------------------------------------------------------
        # 4. Authenticate
        # ---------------------------------------------------------------------

        authenticate()

        # ---------------------------------------------------------------------
        # 5. Search Sentinel-1
        # ---------------------------------------------------------------------

        products = run_search(events)

        # ---------------------------------------------------------------------
        # 6. Save metadata index
        # ---------------------------------------------------------------------

        products = save_index(products)

        # ---------------------------------------------------------------------
        # 7. Summary
        # ---------------------------------------------------------------------

        print_summary(
            events,
            products,
        )

        print()
        print("=" * 80)
        print("COMPLETE")
        print("=" * 80)

        print()
        print("Sentinel-1 metadata:")

        print(INDEX_FILE)

        print()
        print("Raw products:")

        print(RAW_ROOT)

        print()
        print("Next step:")

        print(
            "Extract Sentinel-1 SAR features and merge them "
            "with ERA5 + GDF using timestamp/event windows."
        )

    except KeyboardInterrupt:

        print()
        print("Interrupted by user.")

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
