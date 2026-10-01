"""Daily cloud shell: a second sphere above the Earth carrying that day's VIIRS clouds.

Each 10-degree tile's mask is built from NASA GIBS VIIRS true-colour imagery:
  - composite NOAA-20, then NOAA-21, then SNPP to fill each other's swath gaps
  - cloud = pixels whiter than the cloud-free Blue Marble tile at the same spot (so bright
    deserts and ice sheets cancel out), faded by saturation (so dust and haze don't count)

Masks are cached per date under data/cloud_tiles/YYYY-MM-DD/.

In the shader every tile's mask goes through one shared "Cloud Density" node group:
    mask -> haze clamp -> log(1 + k*mask) / log(1 + k) -> Color Ramp -> alpha
The haze clamp zeroes mask values below a cutoff and rescales the rest to 0..1. The cutoff
is Earth["cloud_haze_ocean"] over water and Earth["cloud_haze_land"] over land (picked by
the Earth's water mask): clear ocean reads slightly brighter in VIIRS than in the Blue
Marble reference, which shows up as false thin cloud unless clamped.
with k = Earth["cloud_thin_boost"] (0 = linear, higher = thin cloud more opaque).

Cloud relief needs a height, and the masks can't supply one directly: thick cloud saturates
at 1.0, so a deck is a flat plateau. A per-date cloud HEIGHT map is therefore derived
offline from the masks: height = mask * blur(mask) over a 3x3 neighbourhood of tiles
(domes over cloud masses, tapered edges, flat decks stay flat; neighbour-aware with a
1-texel overlap so tile edges agree). The shared "Cloud Height" group adds 3D billow
noise (Earth["cloud_billow"]) for texture inside decks.

Cloud relief (shading + shadows), two modes:
  BUMP          Bump node on the normal only. Full mask detail (~2 km/px), shades the cloud
                tops, costs nothing extra. Clouds stay geometrically flat.
  DISPLACEMENT  Bump plus real displacement of the shell by density * Earth["cloud_relief_km"],
                with the shell subdivided further. Cloud tops rise, shadows fall with the
                true height offset, tops show on the horizon. ~16x the cloud geometry at render.
"""

import datetime
import os
from concurrent.futures import ThreadPoolExecutor

import bpy
import numpy as np
import OpenImageIO as oiio

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

# Registered enum (a dropdown), not a custom property: a string custom property shows up as
# a free-text box that does nothing when typed into.
RELIEF_PROP = "phonehome_cloud_relief"
RELIEF_ITEMS = [
    ('BUMP', "Bump (fast)", "Shading on the cloud tops only: full detail, no extra render cost"),
    ('DISPLACEMENT', "Displacement (heavy)",
     "Real cloud height and shadow offset; ~16x cloud geometry, slower renders and more memory"),
]
BRIGHTNESS_PROP = "cloud_brightness"          # cloud base colour (reflectance), 0..1
RELIEF_KM_PROP = "cloud_relief_km"            # displacement height of the densest cloud
BUMP_PROP = "cloud_bump"                      # bump strength 0..1
RELIEF_MODES = ("BUMP", "DISPLACEMENT")
HAZE_OCEAN_PROP = "cloud_haze_ocean"          # haze cutoff over water, 0..1
HAZE_LAND_PROP = "cloud_haze_land"            # haze cutoff over land, 0..1
BILLOW_PROP = "cloud_billow"                  # 3D noise on the cloud height, 0..1
HEIGHT_GROUP = "Cloud Height"
HEIGHT_BLUR_PX = 4                            # box radius; 3 passes ~ gaussian sigma 7 px (~14 km)
# (viewport, render) subdivision of the cloud shell per mode
RELIEF_SUBDIV = {"BUMP": (3, 4), "DISPLACEMENT": (4, 6)}

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


# ---------------------------------------------------------------- derived height tiles (offline)

def cloud_height_path(date, lat0, lon0):
    return config.data_path("cloud_tiles", date.isoformat(), "height", f"height_{lat0:+03d}_{lon0:+04d}.png")


def _blur1d(a, r, axis):
    a = np.moveaxis(a, axis, 0)
    p = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)], 0)
    c = np.cumsum(p, 0, dtype=np.float64)
    c = np.concatenate([np.zeros_like(c[:1]), c], 0)
    return np.moveaxis(((c[2 * r + 1:] - c[:-(2 * r + 1)]) / (2 * r + 1)).astype(np.float32), 0, axis)


def _blur(a, r, passes=3):
    for _ in range(passes):
        a = _blur1d(_blur1d(a, r, 0), r, 1)
    return a


def make_height_tiles(date, mask_paths):
    """Cloud height per tile, derived from the cached masks -- no downloads.

    Each tile is blurred inside its 3x3 neighbourhood (longitude wraps, latitude clamps)
    and written (px + 1) square with the first row/column of its south/east neighbours,
    matching the terrain tiles, so shared edge vertices displace identically.
    """
    paths = {t: cloud_height_path(date, *t) for t in mask_paths}
    missing = [t for t, p in paths.items() if not os.path.exists(p)]
    if not missing:
        return paths
    os.makedirs(os.path.dirname(cloud_height_path(date, 0, 0)), exist_ok=True)
    print(f"Deriving {len(missing)} cloud height tiles for {date.isoformat()} from the masks...")
    step = config.STEP_DEG
    masks = {}

    def mask(lat0, lon0):
        lat0 = min(max(lat0, -90), 90 - step)
        lon0 = (lon0 + 180) % 360 - 180
        if (lat0, lon0) not in masks:
            masks[(lat0, lon0)] = images.read(mask_paths[(lat0, lon0)], oiio.UINT8)[:, :, 0]
        return masks[(lat0, lon0)]

    for lat0, lon0 in missing:
        # rows top->bottom = north->south, cols left->right = west->east
        big = np.vstack([np.hstack([mask(lat0 + dy * step, lon0 + dx * step) for dx in (-1, 0, 1)])
                         for dy in (1, 0, -1)]).astype(np.float32) / 255
        n = big.shape[0] // 3
        h = big * _blur(big, HEIGHT_BLUR_PX)
        tile = h[n:2 * n + 1, n:2 * n + 1]
        images.write_gray8(paths[(lat0, lon0)], (np.clip(tile, 0, 1) * 255 + 0.5).astype(np.uint8))
    return paths


def prepare(date, progress=None):
    """Everything a date needs on disk: masks (fetched or cached) + derived heights."""
    masks = fetch_cloud_tiles(date, progress)
    return {"mask": masks, "height": make_height_tiles(date, masks)}


# ---------------------------------------------------------------- shader

def build_height_group(earth_obj):
    """Shared: derived cloud height in, billowed height out (3D noise in Earth space, km)."""
    ng = bpy.data.node_groups.get(HEIGHT_GROUP)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(HEIGHT_GROUP, "ShaderNodeTree")
    ng.interface.new_socket("Height", in_out='INPUT', socket_type='NodeSocketFloat')
    ng.interface.new_socket("Height", in_out='OUTPUT', socket_type='NodeSocketFloat')
    nodes, links = ng.nodes, ng.links
    gin = nodes.new("NodeGroupInput")
    gin.location = (-800, 0)
    gout = nodes.new("NodeGroupOutput")
    gout.location = (500, 0)

    tc = nodes.new("ShaderNodeTexCoord")
    tc.object = earth_obj               # km in Earth space: noise doesn't swim with shell scale
    tc.location = (-800, -250)
    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-550, -250)
    noise.inputs["Scale"].default_value = 0.03     # ~33 km billows
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.55
    links.new(tc.outputs["Object"], noise.inputs["Vector"])

    billow = nodes.new("ShaderNodeValue")
    billow.name = billow.label = "Billow"
    billow.location = (-550, -500)
    drivers.drive(billow.outputs[0], "default_value", earth_obj,
                  {"b": drivers.prop_path(BILLOW_PROP)}, "min(max(b, 0), 1)")

    def math(op, a, b, x, clamp=False):
        n = nodes.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        n.location = (x, -250)
        for i, v in enumerate((a, b)):
            if hasattr(v, "is_linked"):
                links.new(v, n.inputs[i])
            else:
                n.inputs[i].default_value = v
        return n.outputs[0]

    # height * (1 + billow * (2*noise - 1)), clamped to 0..1
    centred = math('MULTIPLY_ADD', noise.outputs["Fac"], 2.0, -300)
    centred.node.inputs[2].default_value = -1.0
    swing = math('MULTIPLY', centred, billow.outputs[0], -100)
    factor = math('ADD', swing, 1.0, 100)
    out = math('MULTIPLY', gin.outputs["Height"], factor, 300, clamp=True)
    links.new(out, gout.inputs["Height"])
    return ng


def build_density_group(earth_obj):
    ng = bpy.data.node_groups.get(DENSITY_GROUP)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(DENSITY_GROUP, "ShaderNodeTree")
    ng.interface.new_socket("Mask", in_out='INPUT', socket_type='NodeSocketFloat')
    ng.interface.new_socket("Water", in_out='INPUT', socket_type='NodeSocketFloat')
    ng.interface.new_socket("Density", in_out='OUTPUT', socket_type='NodeSocketFloat')
    ng.interface.new_socket("Brightness", in_out='OUTPUT', socket_type='NodeSocketFloat')
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

    # haze clamp: cutoff = mix(land, ocean, water); mask' = clamp((mask - cutoff) / (1 - cutoff))
    cut = {}
    for i, (prop, label) in enumerate(((HAZE_LAND_PROP, "Haze Cutoff Land"), (HAZE_OCEAN_PROP, "Haze Cutoff Ocean"))):
        v = nodes.new("ShaderNodeValue")
        v.name = v.label = label
        v.location = (-1300, -300 - 150 * i)
        drivers.drive(v.outputs[0], "default_value", earth_obj, {"c": drivers.prop_path(prop)},
                      "min(max(c, 0), 0.95)")
        cut[prop] = v.outputs[0]
    cutoff = nodes.new("ShaderNodeMix")
    cutoff.data_type = 'FLOAT'
    cutoff.location = (-1100, -200)
    links.new(gin.outputs["Water"], cutoff.inputs["Factor"])
    links.new(cut[HAZE_LAND_PROP], cutoff.inputs["A"])
    links.new(cut[HAZE_OCEAN_PROP], cutoff.inputs["B"])
    above = nodes.new("ShaderNodeMath")
    above.operation = 'SUBTRACT'
    above.location = (-900, 0)
    links.new(gin.outputs["Mask"], above.inputs[0])
    links.new(cutoff.outputs["Result"], above.inputs[1])
    span = nodes.new("ShaderNodeMath")
    span.operation = 'SUBTRACT'
    span.location = (-900, -200)
    span.inputs[0].default_value = 1.0
    links.new(cutoff.outputs["Result"], span.inputs[1])
    clamped = nodes.new("ShaderNodeMath")
    clamped.operation = 'DIVIDE'
    clamped.use_clamp = True
    clamped.name = clamped.label = "Haze Clamp"
    clamped.location = (-700, 0)
    links.new(above.outputs[0], clamped.inputs[0])
    links.new(span.outputs[0], clamped.inputs[1])
    links.new(clamped.outputs[0], kx.inputs[0])
    links.new(boost.outputs[0], kx.inputs[1])
    links.new(kx.outputs[0], num.inputs[0])
    links.new(boost.outputs[0], base.inputs[0])
    links.new(num.outputs[0], log.inputs[0])
    links.new(base.outputs[0], log.inputs[1])
    links.new(log.outputs[0], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], gout.inputs["Density"])

    brightness = nodes.new("ShaderNodeValue")
    brightness.name = brightness.label = "Cloud Brightness"
    brightness.location = (100, -250)
    drivers.drive(brightness.outputs[0], "default_value", earth_obj,
                  {"v": drivers.prop_path(BRIGHTNESS_PROP)}, "min(max(v, 0), 1)")
    links.new(brightness.outputs[0], gout.inputs["Brightness"])
    return ng


def _cloud_material(lat0, lon0, group, height_group, mask_path, height_path, water_path, earth_obj):
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
    wtex = nodes.new("ShaderNodeTexImage")
    wtex.name = wtex.label = "Water Mask"
    wtex.location = (-400, -150)
    wtex.extension = 'EXTEND'
    wtex.image = bpy.data.images.load(water_path, check_existing=True)   # shared with the Earth tile
    wtex.image.colorspace_settings.name = 'Non-Color'
    links.new(wtex.outputs["Color"], dens.inputs["Water"])
    links.new(dens.outputs["Density"], bsdf.inputs["Alpha"])
    links.new(dens.outputs["Brightness"], bsdf.inputs["Base Color"])

    htex = nodes.new("ShaderNodeTexImage")
    htex.name = htex.label = "Cloud Height"
    htex.location = (-400, -350)
    htex.extension = 'EXTEND'
    htex.interpolation = 'Cubic'
    htex.image = bpy.data.images.load(height_path)
    htex.image.name = mat.name + "_height"
    htex.image.colorspace_settings.name = 'Non-Color'
    hgrp = nodes.new("ShaderNodeGroup")
    hgrp.name = hgrp.label = HEIGHT_GROUP
    hgrp.node_tree = height_group
    hgrp.location = (-150, -350)
    links.new(htex.outputs["Color"], hgrp.inputs["Height"])

    bump = nodes.new("ShaderNodeBump")
    bump.name = bump.label = "Cloud Bump"
    bump.location = (100, -350)
    # Same height scale as the displacement, so bump shades the tops to match it.
    drivers.drive(bump.inputs["Distance"], "default_value", earth_obj,
                  {"h": drivers.prop_path(RELIEF_KM_PROP)}, "h")
    links.new(hgrp.outputs["Height"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    drivers.drive(bump.inputs["Strength"], "default_value", earth_obj,
                  {"b": drivers.prop_path(BUMP_PROP)}, "min(max(b, 0), 1)")

    disp = nodes.new("ShaderNodeDisplacement")
    disp.name = disp.label = "Cloud Displacement"
    disp.location = (100, -550)
    disp.inputs["Midlevel"].default_value = 0.0
    links.new(hgrp.outputs["Height"], disp.inputs["Height"])
    relief = {"h": drivers.prop_path(RELIEF_KM_PROP)}
    drivers.drive(disp.inputs["Scale"], "default_value", earth_obj, relief, "h")
    drivers.drive(mat, "max_vertex_displacement", earth_obj, relief, "h")   # EEVEE culling bounds
    return mat


def _apply_relief(clouds_obj, mode):
    for mat in clouds_obj.data.materials:
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        out = nodes["Material Output"].inputs["Displacement"]
        for link in list(out.links):
            links.remove(link)
        if mode == "DISPLACEMENT":
            links.new(nodes["Cloud Displacement"].outputs["Displacement"], out)
            mat.displacement_method = 'DISPLACEMENT'   # fine detail already comes from the Bump node
        else:
            mat.displacement_method = 'BUMP'
    sub = clouds_obj.modifiers["Subdivide"]
    sub.levels, sub.render_levels = RELIEF_SUBDIV[mode]


def _relief_changed(self, context):
    clouds_obj = bpy.data.objects.get(config.CLOUDS_NAME)
    if clouds_obj is not None and self.name == config.EARTH_NAME:
        _apply_relief(clouds_obj, getattr(self, RELIEF_PROP))


def register_props():
    """The relief dropdown on the Earth. Idempotent; used by the build and by the control panel."""
    if not hasattr(bpy.types.Object, RELIEF_PROP):
        setattr(bpy.types.Object, RELIEF_PROP, bpy.props.EnumProperty(
            name="Cloud Relief", items=RELIEF_ITEMS, default='BUMP', update=_relief_changed))


def set_relief_mode(earth_obj, clouds_obj, mode):
    """Switch every cloud tile between bump-only and real displacement."""
    mode = mode.upper()
    if mode not in RELIEF_MODES:
        raise ValueError(f"cloud relief mode {mode!r}: expected one of {RELIEF_MODES}")
    register_props()
    _apply_relief(clouds_obj, mode)          # explicit: setting an unchanged enum may not fire update
    setattr(earth_obj, RELIEF_PROP, mode)


# ---------------------------------------------------------------- object

def build(earth_obj, date, boost=4.0, relief_mode="BUMP", relief_km=25.0, bump=0.5, billow=0.4,
          haze_ocean=0.2, haze_land=0.0, brightness=0.75):
    """Cloud shell sharing the Earth's mesh layout, UVs and tile order."""
    prepared = prepare(date)

    drivers.set_prop(earth_obj, BOOST_PROP, boost, soft_max=50.0,
                     description="Log curve on every cloud tile's mask: 0 = linear, higher = thin cloud more opaque")
    earth_obj[DATE_PROP] = date.isoformat()
    for prop, val, lo, hi in zip(DATE_PICK_PROPS, (date.year, date.month, date.day), (2012, 1, 1), (2100, 12, 31)):
        earth_obj[prop] = val
        earth_obj.id_properties_ui(prop).update(min=lo, max=hi, description="Cloud imagery date "
                                                "(Load Clouds in Object Properties > PhoneHome Clouds)")
    drivers.set_prop(earth_obj, RELIEF_KM_PROP, relief_km, soft_max=200.0,
                     description="Cloud displacement height of the densest cloud, km (Displacement mode)")
    drivers.set_prop(earth_obj, BUMP_PROP, bump, soft_max=1.0,
                     description="Cloud bump strength: shading detail on the cloud tops")
    earth_obj.id_properties_ui(BUMP_PROP).update(max=1.0, subtype='FACTOR')
    drivers.set_prop(earth_obj, BILLOW_PROP, billow, soft_max=1.0,
                     description="3D noise on the cloud height: texture inside cloud decks")
    earth_obj.id_properties_ui(BILLOW_PROP).update(max=1.0, subtype='FACTOR')
    for prop, val, desc in ((HAZE_OCEAN_PROP, haze_ocean, "Haze clamp over water: mask values below this become clear sky"),
                            (HAZE_LAND_PROP, haze_land, "Haze clamp over land: mask values below this become clear sky")):
        drivers.set_prop(earth_obj, prop, val, desc, soft_max=0.95)
        earth_obj.id_properties_ui(prop).update(max=0.95, subtype='FACTOR')
    drivers.set_prop(earth_obj, BRIGHTNESS_PROP, brightness, soft_max=1.0,
                     description="Cloud brightness (reflectance): 1 = pure white, real cloud tops ~0.6-0.8")
    earth_obj.id_properties_ui(BRIGHTNESS_PROP).update(max=1.0, subtype='FACTOR')
    group = build_density_group(earth_obj)
    height_group = build_height_group(earth_obj)

    mesh = earth_obj.data.copy()
    mesh.name = config.CLOUDS_NAME
    for i, emat in enumerate(earth_obj.data.materials):
        key = (emat["tile_lat0"], emat["tile_lon0"])
        mesh.materials[i] = _cloud_material(*key, group, height_group, prepared["mask"][key],
                                            prepared["height"][key], earth.water_tile_path(*key), earth_obj)

    clouds = bpy.data.objects.new(config.CLOUDS_NAME, mesh)
    for coll in earth_obj.users_collection:
        coll.objects.link(clouds)
    clouds.parent = earth_obj
    clouds.hide_select = True
    earth.add_sphere_modifiers(clouds, *RELIEF_SUBDIV["BUMP"])
    set_relief_mode(earth_obj, clouds, relief_mode)

    exag = {"exag": drivers.prop_path(config.EXAGGERATION_PROP)}
    r = config.RADIUS_KM
    for axis in range(3):
        drivers.drive(clouds, "scale", earth_obj, exag, f"({r} + {shell_altitude_expr()}) / {r}", axis)
    return clouds


def picked_date(earth_obj):
    return datetime.date(*(int(earth_obj[p]) for p in DATE_PICK_PROPS))


def apply(earth_obj, clouds_obj, date, prepared):
    """Swap a different day's masks + heights (from prepare()) onto an existing cloud shell."""
    for mat in clouds_obj.data.materials:
        key = (mat["tile_lat0"], mat["tile_lon0"])
        for node, kind in (("Cloud Mask", "mask"), ("Cloud Height", "height")):
            img = mat.node_tree.nodes[node].image
            img.filepath = prepared[kind][key]
            img.reload()
    earth_obj[DATE_PROP] = date.isoformat()
