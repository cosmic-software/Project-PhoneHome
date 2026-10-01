"""The Earth sphere: locked 10-degree tile grid, NASA colour tiles, terrain displacement.

Geometry is fixed -- derived from RESOLUTION_DEG and impossible to override by
reassigning class attributes or subclassing:

    SEGMENTS = 360 / RESOLUTION_DEG = 36    (10 deg of longitude per segment)
    RINGS    = 180 / RESOLUTION_DEG = 18    (10 deg of latitude per ring)
    vertices = SEGMENTS * (RINGS - 1) + 2 = 614
    faces    = SEGMENTS * RINGS           = 648  (one NASA tile per face)

Each face gets its own material with two images covering exactly that face's box:
  - colour:  NASA GIBS WMS, BlueMarble_ShadedRelief_Bathymetry
  - height:  NASA Visible Earth Blue Marble topography (gebco_08_rev_elev, 21600x10800,
             0 = sea level, 255 = 6400 m), cut into per-face tiles locally

Height drives a Displacement node (Displacement and Bump) whose Scale is driven by
Earth["terrain_exaggeration"] (1 = true relief). A Simple Subdivision + Cast modifier
pair gives the displacement geometry to move while the base mesh stays at 614 verts.

Oceans: a per-tile water mask (sea-level height AND blue-dominant colour, derived offline
from the colour + terrain tiles) feeds the shared "Ocean" node group. Earth["ocean_water"]
slides every tile between the bathymetric Blue Marble ocean (0) and a flat water shader
with sun glint (1); its colour is the group's "Ocean Colour" node.

The mesh is Earth-fixed: +X through (lat 0, lon 0), +Y through (lat 0, lon 90E), +Z north.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import bpy
import numpy as np
import OpenImageIO as oiio
from mathutils import Vector

from . import config, drivers, images, net

GIBS_LAYER = "BlueMarble_ShadedRelief_Bathymetry"

HEIGHT_URL = ("https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/"
              "gebco_08_rev_elev_21600x10800.png")
HEIGHT_WORLD_FILE = "gebco_08_rev_elev_21600x10800.png"
HEIGHT_MAX_KM = 6.4          # pixel value 255 in the NASA topography map

OCEAN_GROUP = "Ocean"
OCEAN_WATER_PROP = "ocean_water"          # 0 = bathymetric map, 1 = water shader
OCEAN_ROUGHNESS_PROP = "ocean_roughness"
OCEAN_COLOUR_DEFAULT = (0.0015, 0.005, 0.022)   # linear RGB; deep open ocean seen from orbit
LAND_ROUGHNESS = 0.8


class _Frozen(type):
    def __setattr__(cls, name, value):
        raise AttributeError(f"{cls.__name__}.{name} is fixed and cannot be changed")


class EarthSphere(metaclass=_Frozen):
    RESOLUTION_DEG = config.STEP_DEG
    SEGMENTS = 360 // RESOLUTION_DEG
    RINGS = 180 // RESOLUTION_DEG
    VERTEX_COUNT = SEGMENTS * (RINGS - 1) + 2
    FACE_COUNT = SEGMENTS * RINGS
    RADIUS_KM = config.RADIUS_KM
    NAME = config.EARTH_NAME
    SUBDIV_VIEWPORT = 4     # 64k faces in the viewport
    SUBDIV_RENDER = 6       # 2.65M faces at render, ~17 km vertex spacing

    def __init_subclass__(cls, **kwargs):
        raise TypeError("EarthSphere cannot be subclassed")

    @classmethod
    def build(cls, tile_px=512, exaggeration=20.0, ocean_water=0.0, ocean_roughness=0.3,
              ocean_colour=OCEAN_COLOUR_DEFAULT):
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=cls.SEGMENTS,
            ring_count=cls.RINGS,
            radius=cls.RADIUS_KM,
            location=(0.0, 0.0, 0.0),
        )
        obj = bpy.context.active_object
        obj.name = cls.NAME
        obj.data.name = cls.NAME
        bpy.ops.object.shade_smooth()

        mesh = obj.data
        if len(mesh.vertices) != cls.VERTEX_COUNT or len(mesh.polygons) != cls.FACE_COUNT:
            raise RuntimeError(f"Earth mesh has {len(mesh.vertices)} verts / {len(mesh.polygons)} faces, "
                               f"expected {cls.VERTEX_COUNT} / {cls.FACE_COUNT}")
        _check_grid_alignment(mesh, cls.RESOLUTION_DEG)

        drivers.set_prop(obj, config.EXAGGERATION_PROP, exaggeration, soft_max=100.0,
                         description=f"Terrain height multiplier (1 = true relief, {HEIGHT_MAX_KM} km max)")

        drivers.set_prop(obj, OCEAN_WATER_PROP, ocean_water, soft_max=1.0,
                         description="Oceans: 0 = bathymetric map, 1 = water shader")
        obj.id_properties_ui(OCEAN_WATER_PROP).update(max=1.0, subtype='FACTOR')
        drivers.set_prop(obj, OCEAN_ROUGHNESS_PROP, ocean_roughness, soft_max=1.0,
                         description="Water shader roughness: lower = tighter, brighter sun glint")

        colour = fetch_colour_tiles(tile_px)
        height = cut_height_tiles()
        water = make_water_tiles(colour, height)
        ocean = build_ocean_group(obj, ocean_colour)
        _assign_tiles(obj, colour, height, water, ocean)
        add_sphere_modifiers(obj, cls.SUBDIV_VIEWPORT, cls.SUBDIV_RENDER)
        return obj


def lat_lon(co):
    v = Vector(co).normalized()
    return math.degrees(math.asin(max(-1.0, min(1.0, v.z)))), math.degrees(math.atan2(v.y, v.x))


def _check_grid_alignment(mesh, step):
    # Every non-pole vertex must sit on a multiple of `step` in both lat and lon,
    # otherwise the faces won't match the tile boxes.
    for v in mesh.vertices:
        lat, lon = lat_lon(v.co)
        if abs(abs(lat) - 90) < 1e-6:
            continue
        for val in (lat, lon):
            if abs(val / step - round(val / step)) > 1e-4:
                raise RuntimeError(f"vertex at lat {lat:.4f}, lon {lon:.4f} is off the {step} deg grid")


def add_sphere_modifiers(obj, levels, render_levels):
    """Simple subdivision + Cast back onto the true sphere.

    Simple (not Catmull-Clark) so each face's UVs stay linear across its tile; Catmull-Clark
    would also shrink a 10-degree sphere by tens of km, more than the terrain itself.
    """
    sub = obj.modifiers.new("Subdivide", 'SUBSURF')
    sub.subdivision_type = 'SIMPLE'
    sub.levels = levels
    sub.render_levels = render_levels
    sub.uv_smooth = 'NONE'

    cast = obj.modifiers.new("Spherify", 'CAST')
    cast.cast_type = 'SPHERE'
    cast.factor = 1.0
    cast.size = config.RADIUS_KM


# ---------------------------------------------------------------- colour tiles (GIBS)

def colour_tile_path(lat0, lon0):
    return config.data_path("earth_tiles", f"{GIBS_LAYER}_{lat0:+03d}_{lon0:+04d}.jpg")


def _fetch_colour_tile(lat0, lon0, tile_px):
    path = colour_tile_path(lat0, lon0)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    data = net.gibs_tile(GIBS_LAYER, lat0, lon0, tile_px)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def fetch_colour_tiles(tile_px):
    tiles = config.tiles()
    os.makedirs(config.data_path("earth_tiles"), exist_ok=True)
    missing = [t for t in tiles if not os.path.exists(colour_tile_path(*t))]
    if missing:
        print(f"Downloading {len(missing)} of {len(tiles)} colour tiles from NASA GIBS ({tile_px}px)...")
    net.init()
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda t: _fetch_colour_tile(*t, tile_px), tiles))
    return dict(zip(tiles, paths))


# ---------------------------------------------------------------- height tiles (Visible Earth)

def height_tile_path(lat0, lon0):
    return config.data_path("terrain_tiles", f"gebco_elev_{lat0:+03d}_{lon0:+04d}.png")


def _load_height_world():
    world_path = config.data_path(HEIGHT_WORLD_FILE)
    if not os.path.exists(world_path):
        print(f"Downloading NASA topography map ({HEIGHT_URL})...")
        os.makedirs(config.DATA_DIR, exist_ok=True)
        _, data = net.fetch(HEIGHT_URL, timeout=300)
        tmp = world_path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, world_path)
    world = images.read(world_path, oiio.UINT8)[:, :, 0].copy()
    # Each pole is a single point, but every tile touching it samples a different stretch
    # of the pole row. Flatten those rows so all tiles agree and the pole doesn't crack.
    world[0, :] = int(round(world[0, :].mean()))
    world[-1, :] = int(round(world[-1, :].mean()))
    return world


def cut_height_tiles():
    """Cut the world heightmap into one PNG per tile.

    Tiles are (px + 1) square: each includes the first row/column of its east and
    north neighbours, so the texel at u/v = 1 in one tile is the same texel as u/v = 0
    in the next. Shared edge verts therefore displace identically in both materials
    and adjacent tiles can't open cracks.
    """
    tiles = config.tiles()
    step = config.STEP_DEG
    os.makedirs(config.data_path("terrain_tiles"), exist_ok=True)
    paths = {t: height_tile_path(*t) for t in tiles}
    missing = [t for t, p in paths.items() if not os.path.exists(p)]
    if not missing:
        return paths

    print(f"Cutting {len(missing)} of {len(tiles)} terrain tiles from {HEIGHT_WORLD_FILE}...")
    world = _load_height_world()
    h, w = world.shape
    ppd = w // 360                     # pixels per degree (60)
    n = step * ppd                     # pixels per tile (600)
    for lat0, lon0 in missing:
        y0 = (90 - (lat0 + step)) * ppd
        x0 = (lon0 + 180) * ppd
        rows = np.clip(np.arange(y0, y0 + n + 1), 0, h - 1)
        cols = np.arange(x0, x0 + n + 1) % w          # wraps across the +/-180 seam
        images.write_gray8(paths[(lat0, lon0)], world[np.ix_(rows, cols)])
    return paths


# ---------------------------------------------------------------- water mask tiles (derived, offline)

def water_tile_path(lat0, lon0):
    return config.data_path("water_tiles", f"water_{lat0:+03d}_{lon0:+04d}.png")


def make_water_tiles(colour, height):
    """Water mask per tile, made from tiles already on disk -- no downloads.

    Water = height exactly at sea level (0 in the NASA topography map) AND blue-dominant
    in the Blue Marble colour. The colour test keeps low-lying land (deltas, polders,
    coastal plains, also height 0) as land; the height test keeps blue-looking land
    (lakes above sea level, glaciers) as land.
    """
    tiles = config.tiles()
    os.makedirs(config.data_path("water_tiles"), exist_ok=True)
    paths = {t: water_tile_path(*t) for t in tiles}
    missing = [t for t in tiles if not os.path.exists(paths[t])]
    if missing:
        print(f"Making {len(missing)} of {len(tiles)} water mask tiles from the cached tiles...")
    for key in missing:
        rgb = images.read_rgb(colour[key])
        n = rgb.shape[0]
        h = images.read(height[key], oiio.UINT8)[:, :, 0]
        # height tiles are (600 + 1) px covering the same box; sample them at the colour pixels
        idx = np.round(np.linspace(0, h.shape[0] - 1, n)).astype(int)
        sea_level = h[np.ix_(idx, idx)] == 0
        blue = (rgb[:, :, 2] > rgb[:, :, 0]) & (rgb[:, :, 2] >= rgb[:, :, 1])
        images.write_gray8(paths[key], ((sea_level & blue) * 255).astype(np.uint8))
    return paths


def build_ocean_group(earth_obj, ocean_colour):
    """Shared group: tile colour + water mask in, final colour + roughness out.

    factor = water mask * Earth["ocean_water"]; one slider moves all 648 tiles between the
    bathymetric map and the water shader.
    """
    ng = bpy.data.node_groups.get(OCEAN_GROUP)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(OCEAN_GROUP, "ShaderNodeTree")
    ng.interface.new_socket("Map Colour", in_out='INPUT', socket_type='NodeSocketColor')
    ng.interface.new_socket("Water", in_out='INPUT', socket_type='NodeSocketFloat')
    ng.interface.new_socket("Colour", in_out='OUTPUT', socket_type='NodeSocketColor')
    ng.interface.new_socket("Roughness", in_out='OUTPUT', socket_type='NodeSocketFloat')
    nodes, links = ng.nodes, ng.links
    gin = nodes.new("NodeGroupInput")
    gin.location = (-700, 0)
    gout = nodes.new("NodeGroupOutput")
    gout.location = (500, 0)

    slider = nodes.new("ShaderNodeValue")
    slider.name = slider.label = "Water Shader"
    slider.location = (-700, -250)
    drivers.drive(slider.outputs[0], "default_value", earth_obj,
                  {"w": drivers.prop_path(OCEAN_WATER_PROP)}, "min(max(w, 0), 1)")
    rough = nodes.new("ShaderNodeValue")
    rough.name = rough.label = "Ocean Roughness"
    rough.location = (-700, -400)
    drivers.drive(rough.outputs[0], "default_value", earth_obj,
                  {"r": drivers.prop_path(OCEAN_ROUGHNESS_PROP)}, "r")
    ocean = nodes.new("ShaderNodeRGB")
    ocean.name = ocean.label = "Ocean Colour"
    ocean.location = (-450, 200)
    ocean.outputs[0].default_value = (*ocean_colour, 1.0)

    fac = nodes.new("ShaderNodeMath")
    fac.operation = 'MULTIPLY'
    fac.use_clamp = True
    fac.location = (-450, -150)
    links.new(gin.outputs["Water"], fac.inputs[0])
    links.new(slider.outputs[0], fac.inputs[1])

    col = nodes.new("ShaderNodeMix")
    col.data_type = 'RGBA'
    col.location = (100, 100)
    links.new(fac.outputs[0], col.inputs["Factor"])
    links.new(gin.outputs["Map Colour"], col.inputs["A"])
    links.new(ocean.outputs[0], col.inputs["B"])
    links.new(col.outputs["Result"], gout.inputs["Colour"])

    r = nodes.new("ShaderNodeMix")
    r.data_type = 'FLOAT'
    r.location = (100, -200)
    r.inputs["A"].default_value = LAND_ROUGHNESS
    links.new(fac.outputs[0], r.inputs["Factor"])
    links.new(rough.outputs[0], r.inputs["B"])
    links.new(r.outputs["Result"], gout.inputs["Roughness"])
    return ng


# ---------------------------------------------------------------- materials

def tile_material(lat0, lon0, colour_path, height_path, water_path, ocean_group, earth):
    mat = bpy.data.materials.new(f"Earth_{lat0:+03d}_{lon0:+04d}")
    mat["tile_lat0"], mat["tile_lon0"] = lat0, lon0
    mat.displacement_method = 'BOTH'
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    output = nodes.get("Material Output")
    bsdf.inputs["Roughness"].default_value = LAND_ROUGHNESS

    tex = nodes.new("ShaderNodeTexImage")
    tex.name = tex.label = "Colour"
    tex.location = (-400, 300)
    tex.image = bpy.data.images.load(colour_path, check_existing=True)
    tex.extension = 'EXTEND'   # stop edge pixels wrapping to the opposite side of the tile

    wtex = nodes.new("ShaderNodeTexImage")
    wtex.name = wtex.label = "Water Mask"
    wtex.location = (-400, 0)
    wtex.image = bpy.data.images.load(water_path, check_existing=True)
    wtex.image.colorspace_settings.name = 'Non-Color'
    wtex.extension = 'EXTEND'

    ocean = nodes.new("ShaderNodeGroup")
    ocean.name = ocean.label = OCEAN_GROUP
    ocean.node_tree = ocean_group
    ocean.location = (-150, 200)
    links.new(tex.outputs["Color"], ocean.inputs["Map Colour"])
    links.new(wtex.outputs["Color"], ocean.inputs["Water"])
    links.new(ocean.outputs["Colour"], bsdf.inputs["Base Color"])
    links.new(ocean.outputs["Roughness"], bsdf.inputs["Roughness"])

    htex = nodes.new("ShaderNodeTexImage")
    htex.name = htex.label = "Height"
    htex.location = (-400, -150)
    htex.image = bpy.data.images.load(height_path, check_existing=True)
    htex.image.colorspace_settings.name = 'Non-Color'
    htex.extension = 'EXTEND'

    disp = nodes.new("ShaderNodeDisplacement")
    disp.location = (0, -250)
    disp.inputs["Midlevel"].default_value = 0.0     # black = sea level, oceans stay put
    links.new(htex.outputs["Color"], disp.inputs["Height"])
    links.new(disp.outputs["Displacement"], output.inputs["Displacement"])

    exag = {"exag": drivers.prop_path(config.EXAGGERATION_PROP)}
    expr = f"exag * {HEIGHT_MAX_KM}"
    drivers.drive(disp.inputs["Scale"], "default_value", earth, exag, expr)
    drivers.drive(mat, "max_vertex_displacement", earth, exag, expr)   # EEVEE culling bounds
    return mat


def _assign_tiles(obj, colour, height, water, ocean_group):
    step = config.STEP_DEG
    mesh = obj.data
    uv = mesh.uv_layers.active.data
    mat_index = {}

    for poly in mesh.polygons:
        lat_c, lon_c = lat_lon(poly.center)
        key = (int(math.floor(lat_c / step)) * step, int(math.floor(lon_c / step)) * step)
        lat0, lon0 = key

        if key not in mat_index:
            mat_index[key] = len(mesh.materials)
            mesh.materials.append(tile_material(lat0, lon0, colour[key], height[key], water[key],
                                                ocean_group, obj))
        poly.material_index = mat_index[key]

        # Map each corner into the tile's 0..1 square.
        for li in poly.loop_indices:
            lat, lon = lat_lon(mesh.vertices[mesh.loops[li].vertex_index].co)
            if abs(abs(lat) - 90) < 1e-6:
                u = 0.5   # pole: longitude undefined, pin to tile centre
            else:
                u = (((lon - lon0 + 1) % 360) - 1) / step   # unwrap across the +/-180 seam
            v = (lat - lat0) / step
            uv[li].uv = (min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0))

    if len(mat_index) != EarthSphere.FACE_COUNT:
        raise RuntimeError(f"{len(mat_index)} tiles assigned, expected {EarthSphere.FACE_COUNT}")
