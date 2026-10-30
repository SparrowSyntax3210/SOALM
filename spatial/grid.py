# ============================================================
# SOALM SPATIAL GRID ENGINE
# ============================================================

import math


# ============================================================
# EARTH CONSTANT
# ============================================================

KM_PER_DEGREE_LAT = 111.0


# ============================================================
# GENERATE GRID
# ============================================================

def generate_grid(
    latitude,
    longitude,
    radius_km=5.0,
    rows=5,
    cols=5
):
    """
    Generate a rectangular spatial grid centered around
    latitude / longitude.

    Parameters
    ----------
    latitude : float
        Center latitude.

    longitude : float
        Center longitude.

    radius_km : float
        Distance from center to grid boundary.

    rows : int
        Number of grid rows.

    cols : int
        Number of grid columns.

    Returns
    -------
    list[dict]

    Each point contains:

        {
            "latitude": ...,
            "longitude": ...,
            "row": ...,
            "col": ...
        }
    """

    latitude = float(latitude)
    longitude = float(longitude)

    rows = int(rows)
    cols = int(cols)

    if rows < 1:
        raise ValueError("rows must be >= 1")

    if cols < 1:
        raise ValueError("cols must be >= 1")

    if radius_km <= 0:
        raise ValueError("radius_km must be > 0")

    # --------------------------------------------------------
    # Latitude distance
    # --------------------------------------------------------

    radius_lat = radius_km / KM_PER_DEGREE_LAT

    # --------------------------------------------------------
    # Longitude distance
    #
    # Longitude degrees become smaller as latitude increases.
    # --------------------------------------------------------

    cos_lat = max(
        math.cos(
            math.radians(latitude)
        ),
        0.01
    )

    radius_lon = (
        radius_km /
        (
            KM_PER_DEGREE_LAT *
            cos_lat
        )
    )

    # --------------------------------------------------------
    # Grid boundaries
    # --------------------------------------------------------

    min_lat = latitude - radius_lat
    max_lat = latitude + radius_lat

    min_lon = longitude - radius_lon
    max_lon = longitude + radius_lon

    # --------------------------------------------------------
    # Generate evenly spaced coordinates
    # --------------------------------------------------------

    if rows == 1:
        latitudes = [latitude]
    else:
        lat_step = (
            max_lat - min_lat
        ) / (rows - 1)

        latitudes = [
            min_lat + i * lat_step
            for i in range(rows)
        ]

    if cols == 1:
        longitudes = [longitude]
    else:
        lon_step = (
            max_lon - min_lon
        ) / (cols - 1)

        longitudes = [
            min_lon + j * lon_step
            for j in range(cols)
        ]

    # --------------------------------------------------------
    # Build points
    # --------------------------------------------------------

    points = []

    for row in range(rows):

        for col in range(cols):

            points.append({

                "latitude": round(
                    latitudes[row],
                    6
                ),

                "longitude": round(
                    longitudes[col],
                    6
                ),

                "row": row,

                "col": col
            })

    return points