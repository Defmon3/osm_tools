#!/usr/bin/env python3
"""
GeoJSON Polygon Filtering Module

Built to filter building sizes from an Overpass-Turbo GeoJSON file.

[out:json][timeout:25];
nwr["building"="yes"]["building"!="hospital"]["building"!="residential"]["building"!="school"]({{bbox}});
out geom;

This module provides functionality to filter polygons from a GeoJSON file based on their geometric
properties such as width, height, and optionally area. The primary purpose is to process geographic
data to extract polygons that fit specific dimensional criteria, which can be adjusted with a
hysteresis parameter for greater flexibility. The results are centroids of the filtered polygons,
saved in a CSV format, which can be used for further geographic analysis or mapping applications.

The module leverages libraries such as Fiona for handling GeoJSON files, Pyproj for coordinate
system transformations (specifically to and from UTM zones based on polygon centroids), and Shapely
for geometric operations. Loguru is used for enhanced logging capabilities, providing clear and
configurable output for debugging and process tracking.

Features:
- Load and parse GeoJSON files to extract geometric features.
- Convert geographic coordinates to metric coordinates based on the UTM zone of the centroid.
- Filter polygons by comparing actual dimensions against user-defined limits with hysteresis.
- Optionally filter polygons based on calculated area to exclude anomalies like holes in buildings.
- Output the centroids of qualifying polygons to a CSV file, which can be used in GIS applications.

Usage:
This module is intended to be used as a script with command-line arguments for easy parameter
configuration. It supports filtering by width, height, optional area comparison, and includes
a hysteresis parameter to adjust the strictness of dimension matching. The script outputs a CSV
file listing the latitudes and longitudes of centroids of polygons that meet the filtering criteria.

Exceptions:
- InvalidGeometry: Custom exception for handling geometric transformations that result in invalid shapes.

The module can be extended or integrated into larger Python applications dealing with geographic
data processing or used standalone for specific projects requiring GeoJSON manipulation.

Example command:
python geojson_filter.py --input "path/to/input.geojson" --output "path/to/output.csv" --width 50 --height 50 --area --hysteresis 10

Developed by: [Your Name or Your Organization]
"""

__author__ = "defmon3@github"
__version__ = "1.0.0"
__license__ = "GNUv3"
__website__ = "https://github.com/Defmon3"

from typing import Tuple, List, Any

import fiona
from loguru import logger as log
from pyproj import CRS, Transformer
from shapely import Point
from shapely.geometry import shape
from shapely.ops import transform

crs_wgs = CRS('EPSG:4326')
transformers = {}


class InvalidGeometry(Exception):
    """
    Exception raised for invalid geometries during processing.
    """
    pass


def get_transformer(utm_zone: int) -> Transformer:
    """Get a Pyproj transformer for a specific UTM zone. Only create them once"""
    if utm_zone not in transformers:
        log.debug(f"Creating transformer for UTM zone {utm_zone}")
        transformers.setdefault(utm_zone, Transformer.from_crs(crs_wgs, CRS(f'EPSG:326{utm_zone}'), always_xy=True))
    return transformers[utm_zone]


def determine_utm_zone(lon: float) -> int:
    """Calculate the UTM (Universal Transverse Mercator) zone for a given longitude.

    Args:
        lon (float): Longitude in decimal degrees.

    Returns:
        int: The UTM zone number for the given longitude.
    """
    return int((lon + 180) / 6) + 1


def get_feature(feature: dict) -> Tuple[Point, float, float, float]:
    """Transforms a GeoJSON feature's geometry to UTM and calculates its geometric properties.

        Args:
            feature (dict): A single GeoJSON feature.

        Returns:
            Tuple[Point, float, float, float]: Centroid of the geometry, width, height, and area of the bounding box.

        Raises:
            InvalidGeometry: If the geometry transformation results in an invalid shape.
        """
    geom = shape(feature['geometry'])
    centroid = geom.centroid
    utm_zone = determine_utm_zone(centroid.x)
    transformer = get_transformer(utm_zone)
    geom_transformed = transform(transformer.transform, geom)
    if not geom_transformed.is_valid:
        log.warning("Invalid geometry skipped.")
        raise InvalidGeometry("Invalid geometry")

    min_x, min_y, max_x, maxy = geom_transformed.bounds
    width = max_x - min_x
    height = maxy - min_y
    area = geom_transformed.area
    return centroid, width, height, area


def filter_polygons(
        geojson_input_path: str,
        csv_output_file: str,
        input_width: int,
        input_height: int,
        hysteresis: int,
        compare_area: bool = False
):
    """Filters GeoJSON polygons based on specified dimensions and optionally by area.

        Args:
            geojson_input_path (str): Path to the input GeoJSON file.
            csv_output_file (str): Path to output CSV file where results are saved.
            input_width (int): The target width to filter shapes.
            input_height (int): The target height to filter shapes.
            hysteresis (int): Tolerance added/subtracted to/from width and height during filtering.
            compare_area (bool): Flag to determine if area based filtering should be applied.

        Outputs:
            CSV file with latitude and longitude of centroids of polygons that meet the criteria.
        """

    max_upper: int = max(input_width, input_height) + hysteresis
    max_lower: int = max(max(input_width, input_height) - hysteresis, 0)  # Prevent negative bounds
    min_upper: int = min(input_width, input_height) + hysteresis
    min_lower: int = max(min(input_width, input_height) - hysteresis, 0)  # Prevent negative bounds

    a_min: int = int(min_lower * min_upper * 0.95)
    a_max: int = int(max_upper * max_lower * 1.05)
    output: List = []

    log.info("Loading GeoJSON file...")

    with fiona.open(geojson_input_path) as (input_file):
        log.info(f"Found {len(input_file)} features in the GeoJSON file.")
        log_str = f"Filtering Geojson for a shape with the sizes [{max_upper}-{max_lower}]x[{min_upper}-{min_lower}] with hysteresis {hysteresis}m"
        if compare_area:
            log_str += f" and area with the size: {a_min}-{a_max}m²"
        log.info(log_str)

        for n, feature in enumerate(input_file):
            try:
                centroid, width, height, area = get_feature(feature)
            except InvalidGeometry:
                continue

            if match_geometry(
                    max_upper, min_upper, max_lower, min_lower,
                    area, width, height,
                    compare_area, a_max, a_min
            ):
                log.debug(
                    f"Width: {width}, Height: {height}, Area: {area}, Lat/Lon: {centroid.y}, {centroid.x}")
                output.append((centroid.y, centroid.x))

    log.success(f"Found {len(output)} matching polygons. Writing to CSV...")

    with open(csv_output_file, "w") as f:
        f.write("Latitude,Longitude\n")
        for lat, lon in output:
            f.write(f"{lat},{lon}\n")
    log.success("Filtered GeoJSON has been saved successfully.")


def match_geometry(
        max_upper: int, min_upper: int, max_lower: int, min_lower: int,
        area: Any, width: float, height: float,
        compare_area: bool, a_max: int, a_min: int):
    """
    Evaluates if a given geometry matches specified dimensional and optional area criteria.

    This function checks if the width and height of a geometry fall within the defined maximum and
    minimum bounds, which include a hysteresis factor. If area comparison is enabled, it further
    checks if the area of the geometry falls within the specified minimum and maximum area limits.

    Parameters:
        max_upper (int): Upper boundary for the maximum dimension (width or height).
        min_upper (int): Upper boundary for the minimum dimension (width or height).
        max_lower (int): Lower boundary for the maximum dimension (width or height).
        min_lower (int): Lower boundary for the minimum dimension (width or height).
        area (Any): The area of the geometry. The type should be float or compatible with numerical comparisons.
        width (float): The width of the geometry's bounding box.
        height (float): The height of the geometry's bounding box.
        compare_area (bool): Flag to determine whether to include area comparison in the criteria.
        a_max (int): Maximum allowable area.
        a_min (int): Minimum allowable area.

    Returns:
        bool: True if the geometry matches the specified criteria, False otherwise.

    Example:
        If you have a polygon with width 45, height 50, and area 2200, to check if it fits within
        maximum and minimum dimensions of 50 and 40 respectively, with hysteresis of 5, and
        an area range of 2100 to 2300:

        valid = match_geometry(55, 45, 35, 25, 2200, 45, 50, True, 2300, 2100)
        print(valid)  # Output: True if it fits all criteria, False otherwise.
    """

    if max_upper > max(width, height) > max_lower and min_upper > min(width, height) > min_lower:
        return compare_area and a_max > area > a_min
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Filter GeoJSON polygons by length and width in meter. Outputs a CSV file with the centroids of the filtered polygons.")
    parser.add_argument('-i', '--input', type=str, help='Input GeoJSON file', default='input.geojson')
    parser.add_argument('-o', '--output', type=str, help='Output CSV file', default='output.csv')
    parser.add_argument('-x', '--width', type=int, help='Width in meters', default=50)
    parser.add_argument('-y', '--height', type=int, help='Height in meters', default=50)
    parser.add_argument('-a', '--area', action='store_true',
                        help='Compare with area (used to filter out holes in buildings)', default=50)
    parser.add_argument('--hysteresis', type=int, help='Hysteresis in meters', default=10)
    args = parser.parse_args()

    # profiler = cProfile.Profile()
    # profiler.enable()
    filter_polygons(args.input, args.output, args.width, args.height, args.hysteresis, args.area)
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # stats.print_stats()


if __name__ == '__main__':
    main()
