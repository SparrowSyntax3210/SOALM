import math
import time

import numpy as np

# =============================================================================
# CONFIG
# =============================================================================

KM_PER_DEGREE_LAT = 111.0

DEFAULT_RADIUS_KM = 5.0
DEFAULT_ROWS = 5
DEFAULT_COLS = 5

MIN_GRID_SIZE = 3
MAX_GRID_SIZE = 9

MAX_RADIUS_KM = 20.0


# =============================================================================
# VALIDATION
# =============================================================================


def _validate_grid_request(
    latitude,
    longitude,
    radius_km,
    rows,
    cols,
):
    try:
        latitude = float(latitude)
        longitude = float(longitude)
        radius_km = float(radius_km)
        rows = int(rows)
        cols = int(cols)
    except (TypeError, ValueError):
        raise ValueError("Invalid spatial grid parameters")

    if not np.isfinite(latitude):
        raise ValueError("Invalid latitude")

    if not np.isfinite(longitude):
        raise ValueError("Invalid longitude")

    if not np.isfinite(radius_km):
        raise ValueError("Invalid radius_km")

    if not -90.0 <= latitude <= 90.0:
        raise ValueError("Latitude must be between -90 and 90")

    if not -180.0 <= longitude <= 180.0:
        raise ValueError("Longitude must be between -180 and 180")

    if radius_km <= 0:
        raise ValueError("radius_km must be greater than 0")

    if radius_km > MAX_RADIUS_KM:
        raise ValueError(f"radius_km cannot exceed {MAX_RADIUS_KM} km")

    if not MIN_GRID_SIZE <= rows <= MAX_GRID_SIZE:
        raise ValueError(
            f"rows must be between " f"{MIN_GRID_SIZE} and {MAX_GRID_SIZE}"
        )

    if not MIN_GRID_SIZE <= cols <= MAX_GRID_SIZE:
        raise ValueError(
            f"cols must be between " f"{MIN_GRID_SIZE} and {MAX_GRID_SIZE}"
        )

    return (
        latitude,
        longitude,
        radius_km,
        rows,
        cols,
    )


# =============================================================================
# GENERATE GRID
# =============================================================================


def generate_grid(
    latitude,
    longitude,
    radius_km=DEFAULT_RADIUS_KM,
    rows=DEFAULT_ROWS,
    cols=DEFAULT_COLS,
):
    """
    Generate center points for the spatial grid.

    Example 5x5:

        row 0:  ●  ●  ●  ●  ●
        row 1:  ●  ●  ●  ●  ●
        row 2:  ●  ●  ●  ●  ●
        row 3:  ●  ●  ●  ●  ●
        row 4:  ●  ●  ●  ●  ●

    The complete grid spans:

        latitude  +/- radius_km
        longitude +/- radius_km

    Longitude is corrected according to latitude.
    """

    (
        latitude,
        longitude,
        radius_km,
        rows,
        cols,
    ) = _validate_grid_request(
        latitude,
        longitude,
        radius_km,
        rows,
        cols,
    )

    # -------------------------------------------------------------------------
    # Convert radius from km to degrees
    # -------------------------------------------------------------------------

    radius_lat = radius_km / KM_PER_DEGREE_LAT

    cos_lat = max(
        math.cos(math.radians(latitude)),
        0.01,
    )

    radius_lon = radius_km / (KM_PER_DEGREE_LAT * cos_lat)

    # -------------------------------------------------------------------------
    # Bounding box
    # -------------------------------------------------------------------------

    min_lat = latitude - radius_lat
    max_lat = latitude + radius_lat

    min_lon = longitude - radius_lon
    max_lon = longitude + radius_lon

    # -------------------------------------------------------------------------
    # Grid center coordinates
    # -------------------------------------------------------------------------

    latitudes = np.linspace(
        min_lat,
        max_lat,
        rows,
    )

    longitudes = np.linspace(
        min_lon,
        max_lon,
        cols,
    )

    # -------------------------------------------------------------------------
    # Generate points
    # -------------------------------------------------------------------------

    points = []

    for row in range(rows):

        for col in range(cols):

            points.append(
                {
                    "latitude": round(
                        float(latitudes[row]),
                        6,
                    ),
                    "longitude": round(
                        float(longitudes[col]),
                        6,
                    ),
                    "row": row,
                    "col": col,
                }
            )

    return points


# =============================================================================
# BUILD CELL POLYGON
# =============================================================================


def build_cell_polygon(
    point,
    left=None,
    right=None,
    top=None,
    bottom=None,
):
    """
    Build the geographic rectangle around a grid center.

    Output:

        [
            [south, west],
            [south, east],
            [north, east],
            [north, west],
            [south, west]
        ]

    IMPORTANT:
    Coordinates are [latitude, longitude],
    which is exactly what Leaflet expects.
    """

    latitude = float(point["latitude"])
    longitude = float(point["longitude"])

    # -------------------------------------------------------------------------
    # Calculate horizontal half-size
    # -------------------------------------------------------------------------

    if left is not None:

        half_lon = abs(longitude - float(left["longitude"])) / 2.0

    elif right is not None:

        half_lon = abs(float(right["longitude"]) - longitude) / 2.0

    else:

        half_lon = 0.005

    # -------------------------------------------------------------------------
    # Calculate vertical half-size
    # -------------------------------------------------------------------------

    if top is not None:

        half_lat = abs(latitude - float(top["latitude"])) / 2.0

    elif bottom is not None:

        half_lat = abs(float(bottom["latitude"]) - latitude) / 2.0

    else:

        half_lat = 0.005

    half_lat = max(
        half_lat,
        0.000001,
    )

    half_lon = max(
        half_lon,
        0.000001,
    )

    # -------------------------------------------------------------------------
    # Boundaries
    # -------------------------------------------------------------------------

    south = latitude - half_lat
    north = latitude + half_lat

    west = longitude - half_lon
    east = longitude + half_lon

    # -------------------------------------------------------------------------
    # Return closed polygon
    # -------------------------------------------------------------------------

    return [
        [
            round(south, 6),
            round(west, 6),
        ],
        [
            round(south, 6),
            round(east, 6),
        ],
        [
            round(north, 6),
            round(east, 6),
        ],
        [
            round(north, 6),
            round(west, 6),
        ],
        [
            round(south, 6),
            round(west, 6),
        ],
    ]


# =============================================================================
# BUILD GEOMETRY ZONES
# =============================================================================


def build_zones(
    points,
    rows,
    cols,
):
    """
    Convert grid center points into geographic polygons.

    This function DOES NOT perform ML inference.

    It is purely responsible for:

        point -> polygon
    """

    if not points:
        return []

    rows = int(rows)
    cols = int(cols)

    grid = {}

    for point in points:

        row = int(point["row"])
        col = int(point["col"])

        grid[(row, col)] = point

    zones = []

    for row in range(rows):

        for col in range(cols):

            point = grid.get((row, col))

            if point is None:
                continue

            left = grid.get((row, col - 1))

            right = grid.get((row, col + 1))

            top = grid.get((row - 1, col))

            bottom = grid.get((row + 1, col))

            polygon = build_cell_polygon(
                point=point,
                left=left,
                right=right,
                top=top,
                bottom=bottom,
            )

            zone = {
                "zone_id": f"zone-{row}-{col}",
                "id": f"zone-{row}-{col}",
                "index": row * cols + col,
                "row": row,
                "col": col,
                "center": {
                    "latitude": point["latitude"],
                    "longitude": point["longitude"],
                },
                # Compatibility field.
                "location": {
                    "latitude": point["latitude"],
                    "longitude": point["longitude"],
                },
                # IMPORTANT:
                # This is the actual geometry consumed by Leaflet.
                "coordinates": polygon,
            }

            zones.append(zone)

    return zones


# =============================================================================
# PREDICT ONE SPATIAL POINT
# =============================================================================


def predict_spatial_point(
    point,
    weather,
    models,
    model_metadata,
    features,
    create_features_fn,
    run_models_fn,
    build_risk_summary_fn,
    elevation=0,
):
    """
    Run the existing SOALM model pipeline for one spatial point.
    """

    latitude = float(point["latitude"])
    longitude = float(point["longitude"])

    # -------------------------------------------------------------------------
    # Feature creation
    # -------------------------------------------------------------------------

    X, feature_dict = create_features_fn(
        weather,
        latitude,
        longitude,
        elevation,
    )

    # -------------------------------------------------------------------------
    # Existing SOALM models
    # -------------------------------------------------------------------------

    (
        predictions,
        probabilities,
        model_results,
    ) = run_models_fn(X)

    # -------------------------------------------------------------------------
    # Risk summary
    # -------------------------------------------------------------------------

    summary = build_risk_summary_fn(
        predictions,
        probabilities,
    )

    overall_risk = str(
        summary.get(
            "overall_risk",
            "LOW",
        )
    ).upper()

    return {
        "latitude": latitude,
        "longitude": longitude,
        "row": int(point["row"]),
        "col": int(point["col"]),
        "risk": overall_risk,
        "probability": float(
            summary.get(
                "overall_probability",
                0.0,
            )
            or 0.0
        ),
        "probability_percent": float(
            summary.get(
                "overall_probability_percent",
                0.0,
            )
            or 0.0
        ),
        "flood_probability": float(
            summary.get(
                "flood_probability",
                0.0,
            )
            or 0.0
        ),
        "flood_probability_percent": float(
            summary.get(
                "flood_probability_percent",
                0.0,
            )
            or 0.0
        ),
        "confidence": float(
            summary.get(
                "overall_confidence",
                0.0,
            )
            or 0.0
        ),
        "confidence_percent": float(
            summary.get(
                "overall_confidence_percent",
                0.0,
            )
            or 0.0
        ),
        "risk_by_horizon": summary.get(
            "risk_by_horizon",
            {},
        ),
        "rainfall_forecast": summary.get(
            "rainfall_forecast",
            {},
        ),
        "events": summary.get(
            "events",
            {},
        ),
        "prediction": summary,
        "probabilities": probabilities,
        "model_results": model_results,
        "raw_predictions": predictions,
        "features": feature_dict,
        "feature_vector": feature_dict,
        "feature_count": len(feature_dict),
    }


# =============================================================================
# ATTACH PREDICTIONS TO GEOMETRY
# =============================================================================


def attach_predictions(
    zones,
    predictions,
):
    """
    Merge ML predictions with geographic zones.

    Geometry remains untouched.
    """

    prediction_map = {
        (
            int(prediction["row"]),
            int(prediction["col"]),
        ): prediction
        for prediction in predictions
    }

    result = []

    for zone in zones:

        key = (
            int(zone["row"]),
            int(zone["col"]),
        )

        prediction = prediction_map.get(
            key,
            {},
        )

        merged = dict(zone)

        # ---------------------------------------------------------------------
        # Main risk information
        # ---------------------------------------------------------------------

        merged["risk"] = str(
            prediction.get(
                "risk",
                "LOW",
            )
        ).upper()

        merged["probability"] = float(
            prediction.get(
                "probability",
                0.0,
            )
            or 0.0
        )

        merged["probability_percent"] = float(
            prediction.get(
                "probability_percent",
                0.0,
            )
            or 0.0
        )

        merged["flood_probability"] = float(
            prediction.get(
                "flood_probability",
                0.0,
            )
            or 0.0
        )

        merged["flood_probability_percent"] = float(
            prediction.get(
                "flood_probability_percent",
                0.0,
            )
            or 0.0
        )

        merged["confidence"] = float(
            prediction.get(
                "confidence",
                0.0,
            )
            or 0.0
        )

        merged["confidence_percent"] = float(
            prediction.get(
                "confidence_percent",
                0.0,
            )
            or 0.0
        )

        # ---------------------------------------------------------------------
        # Prediction details
        # ---------------------------------------------------------------------

        merged["risk_by_horizon"] = prediction.get(
            "risk_by_horizon",
            {},
        )

        merged["rainfall_forecast"] = prediction.get(
            "rainfall_forecast",
            {},
        )

        merged["events"] = prediction.get(
            "events",
            {},
        )

        merged["prediction"] = prediction.get(
            "prediction",
            {},
        )

        merged["probabilities"] = prediction.get(
            "probabilities",
            {},
        )

        merged["model_results"] = prediction.get(
            "model_results",
            {},
        )

        merged["raw_predictions"] = prediction.get(
            "raw_predictions",
            {},
        )

        merged["features"] = prediction.get(
            "features",
            {},
        )

        merged["feature_vector"] = prediction.get(
            "feature_vector",
            {},
        )

        merged["feature_count"] = prediction.get(
            "feature_count",
            0,
        )

        result.append(merged)

    return result


# =============================================================================
# SUMMARIZE ZONES
# =============================================================================


def summarize_zones(zones):
    """
    Calculate overall spatial risk.
    """

    risk_order = {
        "LOW": 0,
        "MODERATE": 1,
        "HIGH": 2,
        "CRITICAL": 3,
    }

    distribution = {
        "LOW": 0,
        "MODERATE": 0,
        "HIGH": 0,
        "CRITICAL": 0,
    }

    overall_risk = "LOW"

    highest_risk_zone = None

    for zone in zones:

        risk = str(
            zone.get(
                "risk",
                "LOW",
            )
        ).upper()

        if risk in distribution:
            distribution[risk] += 1

        if risk_order.get(risk, 0) > risk_order.get(overall_risk, 0):
            overall_risk = risk

    if zones:

        highest_risk_zone = max(
            zones,
            key=lambda zone: (
                risk_order.get(
                    str(
                        zone.get(
                            "risk",
                            "LOW",
                        )
                    ).upper(),
                    0,
                ),
                float(
                    zone.get(
                        "probability",
                        0.0,
                    )
                    or 0.0
                ),
            ),
        )

    return {
        "overall_risk": overall_risk,
        "risk_distribution": distribution,
        "highest_risk_zone": highest_risk_zone,
    }


# =============================================================================
# COMPLETE SPATIAL PREDICTION
# =============================================================================


def run_spatial_prediction(
    latitude,
    longitude,
    radius_km,
    rows,
    cols,
    weather_list,
    models,
    model_metadata,
    features,
    create_features_fn,
    run_models_fn,
    build_risk_summary_fn,
    elevation=0,
):
    """
    Complete spatial prediction pipeline.

        coordinates
             ↓
        generate_grid
             ↓
        weather_list
             ↓
        SOALM inference
             ↓
        build_zones
             ↓
        attach_predictions
             ↓
        final response
    """

    start = time.perf_counter()

    (
        latitude,
        longitude,
        radius_km,
        rows,
        cols,
    ) = _validate_grid_request(
        latitude,
        longitude,
        radius_km,
        rows,
        cols,
    )

    # -------------------------------------------------------------------------
    # Generate grid
    # -------------------------------------------------------------------------

    points = generate_grid(
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        rows=rows,
        cols=cols,
    )

    expected_points = rows * cols

    if len(points) != expected_points:
        raise ValueError(
            f"Grid generation failed: "
            f"expected {expected_points} points, "
            f"generated {len(points)}"
        )

    # -------------------------------------------------------------------------
    # Validate weather
    # -------------------------------------------------------------------------

    if not isinstance(weather_list, list):
        raise ValueError("weather_list must be a list")

    if len(weather_list) != len(points):
        raise ValueError(
            "Weather/grid point count mismatch: "
            f"{len(weather_list)} weather responses "
            f"for {len(points)} grid points"
        )

    # -------------------------------------------------------------------------
    # Model predictions
    # -------------------------------------------------------------------------

    predictions = []

    for point, weather in zip(
        points,
        weather_list,
    ):

        result = predict_spatial_point(
            point=point,
            weather=weather,
            models=models,
            model_metadata=model_metadata,
            features=features,
            create_features_fn=create_features_fn,
            run_models_fn=run_models_fn,
            build_risk_summary_fn=build_risk_summary_fn,
            elevation=elevation,
        )

        predictions.append(result)

    # -------------------------------------------------------------------------
    # Build geographic polygons
    # -------------------------------------------------------------------------

    geometry_zones = build_zones(
        points=points,
        rows=rows,
        cols=cols,
    )

    expected_zones = rows * cols

    if len(geometry_zones) != expected_zones:
        raise ValueError(
            f"Zone generation failed: "
            f"expected {expected_zones} zones, "
            f"generated {len(geometry_zones)}"
        )

    # -------------------------------------------------------------------------
    # Attach predictions
    # -------------------------------------------------------------------------

    zones = attach_predictions(
        zones=geometry_zones,
        predictions=predictions,
    )

    # -------------------------------------------------------------------------
    # Validate geometry before returning
    # -------------------------------------------------------------------------

    for zone in zones:

        coordinates = zone.get("coordinates")

        if (
            not isinstance(
                coordinates,
                list,
            )
            or len(coordinates) < 4
        ):

            raise ValueError(f"Invalid geometry for " f"{zone.get('zone_id')}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = summarize_zones(zones)

    elapsed = time.perf_counter() - start

    # -------------------------------------------------------------------------
    # Final response
    # -------------------------------------------------------------------------

    return {
        "success": True,
        "center": {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "elevation": float(elevation),
        },
        "grid": {
            "rows": int(rows),
            "cols": int(cols),
            "radius_km": float(radius_km),
            "points": len(points),
            "zones": len(zones),
            "total_points": len(points),
            "total_zones": len(zones),
        },
        "overall_risk": summary["overall_risk"],
        "risk_distribution": summary["risk_distribution"],
        "highest_risk_zone": summary["highest_risk_zone"],
        "points": predictions,
        "zones": zones,
        "performance": {
            "model_seconds": round(
                elapsed,
                3,
            )
        },
    }


# =============================================================================
# OPTIONAL GEOMETRY-ONLY HELPER
# =============================================================================


def build_spatial_grid(
    latitude,
    longitude,
    radius_km=DEFAULT_RADIUS_KM,
    rows=DEFAULT_ROWS,
    cols=DEFAULT_COLS,
):
    """
    Build a spatial grid without running ML.
    """

    start = time.perf_counter()

    (
        latitude,
        longitude,
        radius_km,
        rows,
        cols,
    ) = _validate_grid_request(
        latitude,
        longitude,
        radius_km,
        rows,
        cols,
    )

    points = generate_grid(
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        rows=rows,
        cols=cols,
    )

    zones = build_zones(
        points=points,
        rows=rows,
        cols=cols,
    )

    return {
        "center": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "configuration": {
            "radius_km": radius_km,
            "rows": rows,
            "cols": cols,
            "total_points": len(points),
            "total_zones": len(zones),
        },
        "points": points,
        "zones": zones,
        "generation_seconds": round(
            time.perf_counter() - start,
            6,
        ),
    }
