def build_zones(points, rows, cols):

    zones = []

    grid = {}

    for point in points:

        grid[
            (
                point["row"],
                point["col"]
            )
        ] = point

    for row in range(rows):

        for col in range(cols):

            point = grid.get(
                (row, col)
            )

            if not point:
                continue

            risk = point["risk"]

            # Determine surrounding cell
            # from neighboring grid points.

            if col < cols - 1:

                right = grid.get(
                    (row, col + 1)
                )

                lon_right = (
                    right["longitude"]
                    if right
                    else point["longitude"]
                )

            else:

                lon_right = point["longitude"]

            if row < rows - 1:

                bottom = grid.get(
                    (row + 1, col)
                )

                lat_bottom = (
                    bottom["latitude"]
                    if bottom
                    else point["latitude"]
                )

            else:

                lat_bottom = point["latitude"]

            if col > 0:

                left = grid.get(
                    (row, col - 1)
                )

                lon_left = (
                    left["longitude"]
                    if left
                    else point["longitude"]
                )

            else:

                lon_left = point["longitude"]

            if row > 0:

                top = grid.get(
                    (row - 1, col)
                )

                lat_top = (
                    top["latitude"]
                    if top
                    else point["latitude"]
                )

            else:

                lat_top = point["latitude"]

            half_lat = abs(
                point["latitude"] -
                lat_top
            ) / 2

            half_lon = abs(
                point["longitude"] -
                lon_left
            ) / 2

            if half_lat == 0:
                half_lat = 0.01

            if half_lon == 0:
                half_lon = 0.01

            polygon = [

                [
                    point["latitude"] - half_lat,
                    point["longitude"] - half_lon
                ],

                [
                    point["latitude"] - half_lat,
                    point["longitude"] + half_lon
                ],

                [
                    point["latitude"] + half_lat,
                    point["longitude"] + half_lon
                ],

                [
                    point["latitude"] + half_lat,
                    point["longitude"] - half_lon
                ],

                [
                    point["latitude"] - half_lat,
                    point["longitude"] - half_lon
                ]
            ]

            zones.append({

                "id":
                    f"{risk.lower()}-cell-{row}-{col}",

                "risk":
                    risk,

                "probability":
                    point["probability"],

                "confidence":
                    point["confidence"],

                "coordinates":
                    polygon
            })

    return zones