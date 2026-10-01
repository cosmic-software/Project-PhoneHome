"""Paths and shared constants."""

import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")        # download / tile cache
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")    # renders (and optional debug .blend)


def set_data_dir(path):
    global DATA_DIR
    DATA_DIR = os.path.abspath(path)


def data_path(*parts):
    return os.path.join(DATA_DIR, *parts)


RADIUS_KM = 6378.1363        # GMAT default Earth EquatorialRadius; 1 Blender unit = 1 km
STEP_DEG = 10                # tile size; every sphere shares the 36 x 18 tile grid

EARTH_NAME = "Earth"
CLOUDS_NAME = "Clouds"
ATMOSPHERE_NAME = "Atmosphere"
SUN_NAME = "Sun"

# Published offline tileset (colour + terrain tiles, plus sample cloud dates).
TILESET_URL = ("https://github.com/cosmic-software/Project-PhoneHome/releases/download/"
               "tiles-v1/phonehome-tiles.zip")

GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"

# Custom property on the Earth that every terrain/cloud/atmosphere driver reads.
EXAGGERATION_PROP = "terrain_exaggeration"


def tiles():
    """(lat0, lon0) of the south-west corner of every tile."""
    return [(lat0, lon0)
            for lat0 in range(-90, 90, STEP_DEG)
            for lon0 in range(-180, 180, STEP_DEG)]
