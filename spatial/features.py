import math
import pandas as pd
import numpy as np


def build_spatial_features(
    weather_response,
    latitude,
    longitude,
    FEATURES
):

    timelines = (
        weather_response
        .get("data", {})
        .get("timelines", [])
    )

    if not timelines:

        raise ValueError(
            "Tomorrow.io returned no timeline data."
        )

    intervals = timelines[0].get(
        "intervals",
        []
    )

    if not intervals:

        raise ValueError(
            "Tomorrow.io returned no intervals."
        )

    rows = []

    for interval in intervals:

        values = interval.get(
            "values",
            {}
        )

        rows.append({

            "time": interval.get(
                "startTime"
            ),

            "temperature_2m":
                values.get("temperature"),

            "relative_humidity_2m":
                values.get("humidity"),

            "dew_point_2m":
                values.get("dewPoint"),

            "precipitation":
                values.get(
                    "precipitationIntensity"
                ),

            "rain":
                values.get(
                    "rainAccumulation"
                ),

            "showers":
                values.get(
                    "rainAccumulation"
                ),

            "cloud_cover":
                values.get(
                    "cloudCover"
                ),

            "surface_pressure":
                values.get(
                    "surfacePressure"
                ),

            "wind_speed_10m":
                values.get(
                    "windSpeed"
                ),

            "wind_direction_10m":
                values.get(
                    "windDirection"
                ),

            "shortwave_radiation":
                0.0
        })

    df = pd.DataFrame(rows)

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    numeric = [
        c for c in df.columns
        if c != "time"
    ]

    for c in numeric:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df[numeric] = (
        df[numeric]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .ffill()
        .fillna(0)
    )

    # --------------------------------------------------
    # Use latest available observation
    # --------------------------------------------------

    i = len(df) - 1

    current = df.iloc[i]

    rain = df["rain"].fillna(0)

    def rolling(hours):

        start = max(
            0,
            i - hours + 1
        )

        return float(
            rain.iloc[start:i + 1].sum()
        )

    rain_1h = rolling(1)
    rain_3h = rolling(3)
    rain_6h = rolling(6)
    rain_12h = rolling(12)
    rain_24h = rolling(24)
    rain_48h = rolling(48)
    rain_72h = rolling(72)

    def change(column, hours):

        if i < hours:

            return 0.0

        return float(
            df[column].iloc[i]
            -
            df[column].iloc[i - hours]
        )

    timestamp = current["time"]

    hour = int(timestamp.hour)
    day_of_week = int(timestamp.dayofweek)
    month = int(timestamp.month)
    day_of_year = int(timestamp.dayofyear)

    hour_sin = math.sin(
        2 * math.pi * hour / 24
    )

    hour_cos = math.cos(
        2 * math.pi * hour / 24
    )

    month_sin = math.sin(
        2 * math.pi * month / 12
    )

    month_cos = math.cos(
        2 * math.pi * month / 12
    )

    day_of_year_sin = math.sin(
        2 * math.pi *
        day_of_year /
        365
    )

    day_of_year_cos = math.cos(
        2 * math.pi *
        day_of_year /
        365
    )

    # Temporary until elevation service
    elevation = 0.0

    values = {

        "latitude": float(latitude),
        "longitude": float(longitude),

        "elevation": elevation,

        "temperature_2m":
            float(current["temperature_2m"]),

        "relative_humidity_2m":
            float(current["relative_humidity_2m"]),

        "dew_point_2m":
            float(current["dew_point_2m"]),

        "precipitation":
            float(current["precipitation"]),

        "rain":
            float(current["rain"]),

        "showers":
            float(current["showers"]),

        "cloud_cover":
            float(current["cloud_cover"]),

        "surface_pressure":
            float(current["surface_pressure"]),

        "wind_speed_10m":
            float(current["wind_speed_10m"]),

        "wind_direction_10m":
            float(current["wind_direction_10m"]),

        "shortwave_radiation":
            float(current["shortwave_radiation"]),

        "rain_last_1h": rain_1h,
        "rain_last_3h": rain_3h,
        "rain_last_6h": rain_6h,
        "rain_last_12h": rain_12h,
        "rain_last_24h": rain_24h,
        "rain_last_48h": rain_48h,
        "rain_last_72h": rain_72h,

        "pressure_change_1h":
            change(
                "surface_pressure",
                1
            ),

        "pressure_change_3h":
            change(
                "surface_pressure",
                3
            ),

        "pressure_change_6h":
            change(
                "surface_pressure",
                6
            ),

        "temperature_change_3h":
            change(
                "temperature_2m",
                3
            ),

        "temperature_change_6h":
            change(
                "temperature_2m",
                6
            ),

        "humidity_change_3h":
            change(
                "relative_humidity_2m",
                3
            ),

        "humidity_change_6h":
            change(
                "relative_humidity_2m",
                6
            ),

        "wind_change_3h":
            change(
                "wind_speed_10m",
                3
            ),

        "wind_change_6h":
            change(
                "wind_speed_10m",
                6
            ),

        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "day_of_year": day_of_year,

        "hour_sin": hour_sin,
        "hour_cos": hour_cos,

        "month_sin": month_sin,
        "month_cos": month_cos,

        "day_of_year_sin":
            day_of_year_sin,

        "day_of_year_cos":
            day_of_year_cos
    }

    X = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    X = (
        X.replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    return X, values