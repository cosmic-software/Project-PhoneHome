"""Daily cloud shell: a second sphere above the Earth carrying that day's VIIRS clouds.

Each 10-degree tile's mask is built from NASA GIBS VIIRS true-colour imagery:
  - composite NOAA-20, then NOAA-21, then SNPP to fill each other's swath gaps
  - cloud = pixels whiter than the cloud-free Blue Marble tile at the same spot (so bright
    deserts and ice sheets cancel out), faded by saturation (so dust and haze don't count)

Masks are cached per date under data/cloud_tiles/YYYY-MM-DD/.

In the shader every tile's mask goes through one shared "Cloud Density" node group:
    mask -> log(1 + k*mask) / log(1 + k) -> Color Ramp -> alpha
with k = Earth["cloud_thin_boost"] (0 = linear, higher = thin cloud more opaque).
"""

import datetime
import os
from concurrent.futures import ThreadPoolExecutor

import bpy
import numpy as np

from . import config, drivers, earth, images, net, tileset

CLOUD_LAYERS = [
    "VIIRS_NOAA20_CorrectedReflectance_TrueColor",
    "VIIRS_NOAA21_CorrectedReflectance_TrueColor",
    "VIIRS_SNPP_CorrectedReflectance_TrueColor",
]
CLOUD_PX = 512
NODATA = 0.06            # RGB sum below this = no satellite pass (swath gap / polar night)

BOOST_PROP = "cloud_thin_boost"
DATE_PROP = "clouds_date"                               # date currently on the shell
DATE_PICK_PROPS = ("cloud_year", "cloud_month", "cloud_day")   # date boxes in the control panel
DENSITY_GROUP = "Cloud Density"

# Shell height above the sphere, per unit of terrain_exaggeration. 8 km clears the 6.4 km
# terrain cap at any exaggeration, so mountains never poke through the clouds.
CLOUD_ALT_KM = 8.0
CLOUD_MIN_ALT_KM = 2.0   # keeps the shell off the sea surface when exaggeration is 0


def cloud_tile_path(date, lat0, lon0):
    return config.data_path("cloud_tiles", date.isoformat(), f"clouds_{lat0:+03d}_{lon0:+04d}.png")


def cached_dates():
    """Dates with a complete set of cloud tiles on disk (usable offline)."""
    return [datetime.date.fromisoformat(d) for d in tileset.cached_cloud_dates()]


def shell_altitude_expr(exag_var="exag"):
    """Driver-expression fragment: cloud shell altitude in km."""
    return f"max({exag_var} * {CLOUD_ALT_KM}, {CLOUD_MIN_ALT_KM})"


# ---------------------------------------------------------------- fetching / masking

def _make_tile(date, lat0, lon0):
    """Build one cloud-mask tile; returns the fraction of the tile with satellite data."""
    sat, hole = None, None
    for layer in CLOUD_LAYERS:
        img = images.decode_rgb(net.gibs_tile(layer, lat0, lon0, CLOUD_PX, date))
        if sat is None:
            sat = img.copy()
        else:
            sat[hole] = img[hole]
        hole = sat.sum(axis=2) < NODATA
        if hole.mean() < 0.002:
            break

    bpath = earth.colour_tile_path(lat0, lon0)
    base = (images.read_rgb(bpath) if os.path.exists(bpath)
            else images.decode_rgb(net.gibs_tile(earth.GIBS_LAYER, lat0, lon0, CLOUD_PX)))
    base = images.resize_nearest(base, CLOUD_PX)

    # Clouds are white: their darkest channel is high. Compare against the same measure on
    # the cloud-free map so sand, salt flats and ice sheets cancel out. Clouds are also
    # grey/white, whereas dust and swath-edge haze are tinted; saturation fades those out.
    # The wide whiteness ramp keeps thin cloud semi-transparent.
    lo, hi = sat.min(axis=2), sat.max(axis=2)
    whiteness = lo - base.min(axis=2)
    saturation = (hi - lo) / np.maximum(hi, 1e-3)
    alpha = (np.clip((whiteness - 0.10) / 0.45, 0.0, 1.0)
             * np.clip((0.40 - saturation) / 0.20, 0.0, 1.0))
    alpha[hole] = 0.0

    images.write_gray8(cloud_tile_path(date, lat0, lon0), (alpha * 255 + 0.5).astype(np.uint8))
    return 1.0 - hole.mean()


def fetch_cloud_tiles(date, progress=None):
    """Download + mask every tile for `date`. Returns {(lat0, lon0): png path}.

    `progress(done, total)` is called as tiles finish (used by the control panel).
    """
    tiles = config.tiles()
    os.makedirs(os.path.dirname(cloud_tile_path(date, 0, 0)), exist_ok=True)
    paths = {t: cloud_tile_path(date, *t) for t in tiles}
    missing = [t for t in tiles if not os.path.exists(paths[t])]
    if not missing:
        return paths

    print(f"Fetching clouds for {date.isoformat()}: {len(missing)} tiles from NASA GIBS (VIIRS)...")
    net.init()
    coverage = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, cov in enumerate(pool.map(lambda t: _make_tile(date, *t), missing), 1):
            coverage.append(cov)
            if progress:
                progress(i, len(missing))
            if i % 65 == 0 or i == len(missing):
                print(f"  clouds {i}/{len(missing)}")

    if sum(coverage) / len(coverage) < 0.05:
        for t in missing:
            os.remove(paths[t])
        raise RuntimeError(f"no VIIRS imagery on GIBS for {date.isoformat()} "
                           f"(NOAA-20 from 2018, NOAA-21 from 2023; very recent days can take a few hours)")
    return paths


# ---------------------------------------------------------------- shader

def build_density_group(earth_obj):
    ng = bpy.data.node_groups.get(DENSITY_GROUP)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(DENSITY_GROUP, "ShaderNodeTree")
    ng.interface.new_socket("Mask", in_out='INPUT', socket_type='NodeSocketFloat')
    ng.interface.new_socket("Density", in_out='OUTPUT', socket_type='NodeSocketFloat')
    nodes, links = ng.nodes, ng.links

    gin = nodes.new("NodeGroupInput")
    gin.location = (-800, 0)
    gout = nodes.new("NodeGroupOutput")
    gout.location = (600, 0)

    boost = nodes.new("ShaderNodeValue")
    boost.name = boost.label = "Thin Cloud Boost"
    boost.location = (-800, -200)
    drivers.drive(boost.outputs[0], "default_value", earth_obj,
                  {"k": drivers.prop_path(BOOST_PROP)}, "max(k, 0.001)")

    def math(op, x, b=None):
        n = nodes.new("ShaderNodeMath")
        n.operation = op
        n.location = (x, 0 if b is None else -200)
        if b is not None:
            n.inputs[1].default_value = b
        return n

    kx = math('MULTIPLY', -550)                  # k * mask
    num = math('ADD', -350, 1.0)                 # 1 + k * mask
    num.location.y = 0
    base = math('ADD', -350, 1.0)                # 1 + k
    log = math('LOGARITHM', -150)                # log base (1+k) of (1 + k*mask): 0..1 -> 0..1
    log.name = log.label = "Log Curve"
    log.use_clamp = True

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.name = ramp.label = "Cloud Ramp"
    ramp.location = (100, 0)
    ramp.color_ramp.elements[0].position = 0.05   # drop the faintest haze
    ramp.color_ramp.elements[1].position = 1.0

    links.new(gin.outputs["Mask"], kx.inputs[0])
    links.new(boost.outputs[0], kx.inputs[1])
    links.new(kx.outputs[0], num.inputs[0])
    links.new(boost.outputs[0], base.inputs[0])
    links.new(num.outputs[0], log.inputs[0])
    links.new(base.outputs[0], log.inputs[1])
    links.new(log.outputs[0], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], gout.inputs["Density"])
    return ng


def _cloud_material(lat0, lon0, group, mask_path):
    mat = bpy.data.materials.new(f"Clouds_{lat0:+03d}_{lon0:+04d}")
    mat["tile_lat0"], mat["tile_lon0"] = lat0, lon0
    if hasattr(mat, "use_transparent_shadow"):
        mat.use_transparent_shadow = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0

    tex = nodes.new("ShaderNodeTexImage")
    tex.name = tex.label = "Cloud Mask"
    tex.location = (-400, 0)
    tex.extension = 'EXTEND'
    tex.interpolation = 'Cubic'   # 'Smart' only differs from Linear under Cycles OSL
    tex.image = bpy.data.images.load(mask_path)
    tex.image.name = mat.name
    tex.image.colorspace_settings.name = 'Non-Color'

    dens = nodes.new("ShaderNodeGroup")
    dens.name = dens.label = DENSITY_GROUP
    dens.node_tree = group
    dens.location = (-150, -100)
    links.new(tex.outputs["Color"], dens.inputs["Mask"])
    links.new(dens.outputs["Density"], bsdf.inputs["Alpha"])
    return mat


# ---------------------------------------------------------------- object

def build(earth_obj, date, boost=4.0):
    """Cloud shell sharing the Earth's mesh layout, UVs and tile order."""
    paths = fetch_cloud_tiles(date)

    drivers.set_prop(earth_obj, BOOST_PROP, boost, soft_max=50.0,
                     description="Log curve on every cloud tile's mask: 0 = linear, higher = thin cloud more opaque")
    earth_obj[DATE_PROP] = date.isoformat()
    for prop, val, lo, hi in zip(DATE_PICK_PROPS, (date.year, date.month, date.day), (2012, 1, 1), (2100, 12, 31)):
        earth_obj[prop] = val
        earth_obj.id_properties_ui(prop).update(min=lo, max=hi, description="Cloud imagery date "
                                                "(Load Clouds in Object Properties > PhoneHome Clouds)")
    group = build_density_group(earth_obj)

    mesh = earth_obj.data.copy()
    mesh.name = config.CLOUDS_NAME
    for i, emat in enumerate(earth_obj.data.materials):
        key = (emat["tile_lat0"], emat["tile_lon0"])
        mesh.materials[i] = _cloud_material(*key, group, paths[key])

    clouds = bpy.data.objects.new(config.CLOUDS_NAME, mesh)
    for coll in earth_obj.users_collection:
        coll.objects.link(clouds)
    clouds.parent = earth_obj
    clouds.hide_select = True
    earth.add_sphere_modifiers(clouds, 3, 4)

    exag = {"exag": drivers.prop_path(config.EXAGGERATION_PROP)}
    r = config.RADIUS_KM
    for axis in range(3):
        drivers.drive(clouds, "scale", earth_obj, exag, f"({r} + {shell_altitude_expr()}) / {r}", axis)
    return clouds


def picked_date(earth_obj):
    return datetime.date(*(int(earth_obj[p]) for p in DATE_PICK_PROPS))


def apply(earth_obj, clouds_obj, date, paths):
    """Swap a different day's masks onto an existing cloud shell."""
    for mat in clouds_obj.data.materials:
        img = mat.node_tree.nodes["Cloud Mask"].image
        img.filepath = paths[(mat["tile_lat0"], mat["tile_lon0"])]
        img.reload()
    earth_obj[DATE_PROP] = date.isoformat()
