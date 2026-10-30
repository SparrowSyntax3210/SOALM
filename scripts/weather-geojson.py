import json
import math
import time
from pathlib import Path
from datetime import datetime, timezone

import requests
from shapely.geometry import Point, shape

# ============================================================
# SANKET - OFFLINE WEATHER CACHE GENERATOR
# ============================================================

GRID_SIZE_KM = 20
HOURS_TO_KEEP = 10

BATCH_SIZE = 100

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

BASE_DIR = Path(__file__).resolve().parent

BOUNDARY_FILE = BASE_DIR / "india.geojson"
OUTPUT_FILE = BASE_DIR / "weather.json"


# ============================================================
# WEATHER VARIABLES
# ============================================================

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "weather_code",
]


# ============================================================
# LOAD INDIA BOUNDARY
# ============================================================


def load_india_boundary():

    print("Loading India boundary...")

    with open(BOUNDARY_FILE, "r", encoding="utf-8") as file:

        geojson = json.load(file)

    # Supports FeatureCollection
    if geojson["type"] == "FeatureCollection":

        geometries = [feature["geometry"] for feature in geojson["features"]]

        return [shape(geometry) for geometry in geometries]

    # Supports single Feature
    elif geojson["type"] == "Feature":

        return [shape(geojson["geometry"])]

    # Supports direct Polygon / MultiPolygon
    else:

        return [shape(geojson)]


# ============================================================
# GENERATE 20 KM GRID
# ============================================================


def generate_india_grid(boundaries):

    print("Generating 20 km grid...")

    # Approximate India bounding box
    min_lat = 8.0
    max_lat = 37.0

    min_lon = 68.0
    max_lon = 97.5

    lat_step = GRID_SIZE_KM / 111.0

    grid = []

    grid_id = 0

    lat = min_lat

    while lat <= max_lat:

        # Longitude distance changes with latitude
        lon_step = GRID_SIZE_KM / (111.0 * math.cos(math.radians(lat)))

        lon = min_lon

        while lon <= max_lon:

            point = Point(lon, lat)

            # Check whether point lies inside India
            inside_india = any(boundary.contains(point) for boundary in boundaries)

            if inside_india:

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

    print(f"Generated {len(grid)} India weather cells.")

    return grid


# ============================================================
# FETCH WEATHER FOR A BATCH
# ============================================================


def fetch_weather_batch(batch):

    latitudes = [str(item["latitude"]) for item in batch]

    longitudes = [str(item["longitude"]) for item in batch]

    params = {
        "latitude": ",".join(latitudes),
        "longitude": ",".join(longitudes),
        "hourly": ",".join(WEATHER_VARIABLES),
        "forecast_days": 1,
        "timezone": "UTC",
    }

    response = requests.get(OPEN_METEO_URL, params=params, timeout=60)

    response.raise_for_status()

    return response.json()


# ============================================================
# PROCESS WEATHER RESPONSE
# ============================================================


def process_weather(grid, weather_response):

    hourly = weather_response["hourly"]

    times = hourly["time"]

    total_hours = len(times)

    # Keep last 10 hours
    start_index = max(0, total_hours - HOURS_TO_KEEP)

    records = []

    for i in range(start_index, total_hours):

        record = {
            "time": times[i],
            "temperature": (hourly["temperature_2m"][i]),
            "humidity": (hourly["relative_humidity_2m"][i]),
            "precipitation": (hourly["precipitation"][i]),
            "windSpeed": (hourly["wind_speed_10m"][i]),
            "weatherCode": (hourly["weather_code"][i]),
        }

        records.append(record)

    return {
        "gridId": grid["gridId"],
        "latitude": grid["latitude"],
        "longitude": grid["longitude"],
        "weather": records,
    }


# ============================================================
# GENERATE WEATHER.JSON
# ============================================================


def generate_weather_json():

    print("\n======================================")
    print(" SANKET WEATHER CACHE GENERATOR")
    print("======================================\n")

    # ----------------------------------
    # Load boundary
    # ----------------------------------

    boundaries = load_india_boundary()

    # ----------------------------------
    # Generate grid
    # ----------------------------------

    grid = generate_india_grid(boundaries)

    weather_cache = {
        "metadata": {
            "application": "Sanket",
            "source": "Open-Meteo",
            "gridSizeKm": GRID_SIZE_KM,
            "hoursStored": HOURS_TO_KEEP,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "totalGrids": len(grid),
        },
        "grids": {},
    }

    # ----------------------------------
    # Process batches
    # ----------------------------------

    total = len(grid)

    for start in range(0, total, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total)

        batch = grid[start:end]

        batch_number = (start // BATCH_SIZE) + 1

        total_batches = math.ceil(total / BATCH_SIZE)

        print(f"\nBatch {batch_number}/{total_batches}")

        try:

            response = fetch_weather_batch(batch)

            # Open-Meteo returns a list
            # when multiple coordinates are requested
            if isinstance(response, list):

                responses = response

            else:

                responses = [response]

            for grid_item, weather_response in zip(batch, responses):

                processed = process_weather(grid_item, weather_response)

                weather_cache["grids"][grid_item["gridId"]] = processed

        except Exception as error:

            print("Batch failed:", error)

        # Small delay between batches
        time.sleep(0.5)

    # ----------------------------------
    # Save JSON
    # ----------------------------------

    print("\nWriting weather.json...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(weather_cache, file, indent=2, ensure_ascii=False)

    print("\n======================================")
    print(" weather.json CREATED")
    print("======================================")

    print(f"Location: {OUTPUT_FILE}")

    print(f"Grid cells: {len(weather_cache['grids'])}")

    print(f"Hours per grid: {HOURS_TO_KEEP}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    generate_weather_json()
