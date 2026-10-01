"""Command line for the headless pipeline (arguments after Blender's `--`)."""

import argparse
import datetime
import os
import sys
import time

from . import atmosphere, clouds, config, earth, net, scene


def _yesterday_utc():
    return datetime.datetime.now(datetime.timezone.utc).date() - datetime.timedelta(days=1)


def _srgb_hex_to_linear(hexcode):
    h = hexcode.lstrip("#")
    if len(h) != 6:
        raise SystemExit(f"--ocean-colour {hexcode}: expected RRGGBB")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(
        prog="run_phonehome.py",
        description="Build and render a NASA-data Earth in headless Blender.")

    out = p.add_argument_group("output")
    out.add_argument("--render", help="PNG path (default: output/phonehome_<clouds date>.png)")
    out.add_argument("--no-render", action="store_true", help="build only, don't render")
    out.add_argument("--save-blend", help="also save the scene as a .blend with the control panel "
                                          "(the project one is blender/phonehome.blend)")
    out.add_argument("--resolution", type=int, default=1024, help="square render size in px (default 1024)")
    out.add_argument("--engine", choices=["BLENDER_EEVEE", "CYCLES"], default="BLENDER_EEVEE")
    out.add_argument("--samples", type=int, help="render samples (engine default if omitted)")
    out.add_argument("--data-dir", help=f"download/tile cache (default: {config.DATA_DIR})")
    out.add_argument("--offline", action="store_true",
                     help="never touch the network; everything must come from the installed tileset")

    e = p.add_argument_group("earth")
    e.add_argument("--exaggeration", type=float, default=20.0,
                   help="terrain height multiplier; 1 = true relief (default 20)")
    e.add_argument("--tile-px", type=int, default=512, help="colour tile size in px (default 512)")
    e.add_argument("--ocean-water", type=float, default=0.0,
                   help="oceans: 0 = bathymetric map, 1 = water shader (default 0)")
    e.add_argument("--ocean-roughness", type=float, default=0.3,
                   help="water shader roughness; lower = tighter sun glint (default 0.3)")
    e.add_argument("--ocean-colour", default=None, metavar="RRGGBB",
                   help="water shader colour as an sRGB hex code (default: deep ocean blue)")

    c = p.add_argument_group("clouds")
    c.add_argument("--clouds", type=datetime.date.fromisoformat, default=None, metavar="YYYY-MM-DD",
                   help="date of the VIIRS cloud imagery (default: yesterday UTC)")
    c.add_argument("--no-clouds", action="store_true")
    c.add_argument("--cloud-boost", type=float, default=4.0,
                   help="log curve on thin cloud: 0 = linear, higher = more opaque (default 4)")
    c.add_argument("--cloud-relief", choices=["bump", "displacement"], default="bump",
                   help="bump: shading only, fast (default); displacement: real cloud height + "
                        "shadow offset, ~16x cloud geometry, slower")
    c.add_argument("--cloud-relief-km", type=float, default=25.0,
                   help="displacement height of the densest cloud in km (default 25)")
    c.add_argument("--cloud-bump", type=float, default=0.5, help="bump strength 0..1 (default 0.5)")
    c.add_argument("--cloud-brightness", type=float, default=0.75,
                   help="cloud reflectance 0..1: 1 = pure white, real cloud tops ~0.6-0.8 (default 0.75)")
    c.add_argument("--haze-ocean", type=float, default=0.2,
                   help="haze clamp over water: mask values below this become clear sky (default 0.2)")
    c.add_argument("--haze-land", type=float, default=0.0,
                   help="haze clamp over land (default 0)")
    c.add_argument("--cloud-billow", type=float, default=0.4,
                   help="3D noise texture on cloud tops 0..1 (default 0.4)")

    a = p.add_argument_group("atmosphere")
    a.add_argument("--no-atmosphere", action="store_true")
    for name, (default, _, desc) in atmosphere.DEFAULTS.items():
        flag = "--" + name.replace("atmosphere_", "atmo-").replace("_", "-")
        a.add_argument(flag, type=float, default=default, dest=name, help=f"{desc} (default {default})")

    s = p.add_argument_group("sun and camera")
    s.add_argument("--sun-lat", type=float, default=10.0, help="sub-solar latitude (default 10)")
    s.add_argument("--sun-lon", type=float, default=70.0, help="sub-solar longitude (default 70)")
    s.add_argument("--sun-strength", type=float, default=4.0, help="sun lamp strength (default 4)")
    s.add_argument("--view-lat", type=float, default=20.0, help="camera latitude (default 20)")
    s.add_argument("--view-lon", type=float, default=0.0, help="camera longitude (default 0)")
    s.add_argument("--view-dist", type=float, default=3.5, help="camera distance in Earth radii (default 3.5)")
    s.add_argument("--lens", type=float, default=50.0, help="camera focal length in mm (default 50)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    if args.data_dir:
        config.set_data_dir(args.data_dir)
    net.OFFLINE = args.offline
    date = args.clouds or _yesterday_utc()
    if date >= datetime.datetime.now(datetime.timezone.utc).date():
        raise SystemExit(f"--clouds {date}: pick a date before today (UTC); today's imagery isn't complete yet")
    if args.offline:
        from . import tileset   # plain Python; checks the cache without touching the network
        st = tileset.status()
        if not st["offline_ready"]:
            raise SystemExit(f"--offline: tileset not installed in {config.DATA_DIR} "
                             f"({st['colour_tiles']} colour / {st['terrain_tiles']} terrain tiles of "
                             f"{st['tiles_expected']}); run: python -m phonehome.tileset install <zip or URL>")
        if not args.no_clouds and date.isoformat() not in st["cloud_dates"]:
            raise SystemExit(f"--offline: no clouds for {date} in the tileset; available: "
                             f"{', '.join(st['cloud_dates']) or 'none'} (or use --no-clouds)")

    scene.clear()
    ocean_colour = _srgb_hex_to_linear(args.ocean_colour) if args.ocean_colour else earth.OCEAN_COLOUR_DEFAULT
    earth_obj = earth.EarthSphere.build(tile_px=args.tile_px, exaggeration=args.exaggeration,
                                        ocean_water=args.ocean_water, ocean_roughness=args.ocean_roughness,
                                        ocean_colour=ocean_colour)
    sun = scene.add_sun(args.sun_lat, args.sun_lon, args.sun_strength)
    print(f"Earth: {len(earth_obj.data.vertices)} verts, {len(earth_obj.data.materials)} tiles, "
          f"terrain x{args.exaggeration}")

    if not args.no_clouds:
        clouds.build(earth_obj, date, boost=args.cloud_boost, relief_mode=args.cloud_relief,
                     relief_km=args.cloud_relief_km, bump=args.cloud_bump, billow=args.cloud_billow,
                     haze_ocean=args.haze_ocean, haze_land=args.haze_land, brightness=args.cloud_brightness)
        print(f"Clouds: {date.isoformat()} ({args.cloud_relief})")
    if not args.no_atmosphere:
        atmosphere.build(earth_obj, sun, {k: getattr(args, k) for k in atmosphere.DEFAULTS})
        print(f"Atmosphere: lit by '{sun.name}'")

    scene.add_camera(args.view_lat, args.view_lon, args.view_dist, args.lens)
    scene.add_world()

    if not args.no_render:
        path = args.render or os.path.join(config.OUTPUT_DIR, f"phonehome_{date.isoformat()}.png")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        scene.render(path, args.engine, args.resolution, args.samples)
        print(f"Rendered {path}")
    if args.save_blend:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_blend)), exist_ok=True)
        scene.save_blend(args.save_blend)
        print(f"Saved {args.save_blend}")
    print(f"Done in {time.time() - t0:.0f}s")
