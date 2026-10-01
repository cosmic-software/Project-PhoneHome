"""Offline tileset: pack the downloaded NASA tiles into one zip, install it anywhere.

With the tileset installed, PhoneHome builds and renders with no network access
(run with --offline to guarantee it). Clouds work offline for any dates in the pack.

Plain Python, no Blender needed:

    python -m phonehome.tileset status
    python -m phonehome.tileset pack    phonehome-tiles.zip [--clouds 2026-09-29 2026-09-28 ...]
    python -m phonehome.tileset install                          (the published tileset)
    python -m phonehome.tileset install phonehome-tiles.zip      (or any https:// URL)
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

from . import config, net

FORMAT = 1
MANIFEST = "phonehome_tileset.json"
BASE_PARTS = ("earth_tiles", "terrain_tiles")


def _count(subdir, ext):
    d = config.data_path(subdir)
    return len([f for f in os.listdir(d) if f.endswith(ext)]) if os.path.isdir(d) else 0


def cached_cloud_dates():
    """ISO dates with a complete set of cloud tiles on disk (usable offline)."""
    root = config.data_path("cloud_tiles")
    if not os.path.isdir(root):
        return []
    n = len(config.tiles())
    out = []
    for name in sorted(os.listdir(root)):
        try:
            datetime.date.fromisoformat(name)
        except ValueError:
            continue
        if len([f for f in os.listdir(os.path.join(root, name)) if f.endswith(".png")]) == n:
            out.append(name)
    return out


def status():
    n = len(config.tiles())
    colour, terrain = _count("earth_tiles", ".jpg"), _count("terrain_tiles", ".png")
    return {
        "data_dir": config.DATA_DIR,
        "tiles_expected": n,
        "colour_tiles": colour,
        "terrain_tiles": terrain,
        "cloud_dates": cached_cloud_dates(),
        "offline_ready": colour == n and terrain == n,
    }


def pack(out_path, cloud_dates=()):
    st = status()
    if not st["offline_ready"]:
        raise SystemExit(f"tile cache incomplete ({st['colour_tiles']} colour, {st['terrain_tiles']} terrain of "
                         f"{st['tiles_expected']}); run PhoneHome once online first")
    dates = [d.isoformat() if isinstance(d, datetime.date) else d for d in cloud_dates]
    missing = [d for d in dates if d not in st["cloud_dates"]]
    if missing:
        raise SystemExit(f"no complete cloud tiles cached for {', '.join(missing)}; "
                         f"cached: {', '.join(st['cloud_dates']) or 'none'}")

    parts = list(BASE_PARTS) + [os.path.join("cloud_tiles", d) for d in dates]
    manifest = {
        "format": FORMAT,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "tiles": st["tiles_expected"],
        "step_deg": config.STEP_DEG,
        "parts": [p.replace(os.sep, "/") for p in parts],
        "cloud_dates": dates,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    files = 0
    # JPEG/PNG are already compressed: store, don't deflate.
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as z:
        z.writestr(MANIFEST, json.dumps(manifest, indent=2))
        for part in parts:
            src = config.data_path(part)
            for name in sorted(os.listdir(src)):
                z.write(os.path.join(src, name), f"{part.replace(os.sep, '/')}/{name}")
                files += 1
    size = os.path.getsize(out_path) / 1e6
    print(f"Packed {files} tiles ({size:.0f} MB) -> {out_path}")
    print(f"  base tiles + clouds for: {', '.join(dates) or '(no cloud dates)'}")
    return out_path


def _download(url, dest):
    if net.OFFLINE:
        raise net.OfflineError(f"offline mode: can't download {url}")
    net.init()
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=120, context=net._ssl_ctx) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def install(src=None):
    src = src or config.TILESET_URL
    tmp = None
    if src.startswith(("http://", "https://")):
        fd, tmp = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        _download(src, tmp)
        src = tmp
    try:
        with zipfile.ZipFile(src) as z:
            try:
                manifest = json.loads(z.read(MANIFEST))
            except KeyError:
                raise SystemExit(f"{src} is not a PhoneHome tileset (no {MANIFEST})")
            if manifest.get("format") != FORMAT or manifest.get("step_deg") != config.STEP_DEG:
                raise SystemExit(f"tileset format {manifest.get('format')} / step {manifest.get('step_deg')} "
                                 f"doesn't match this PhoneHome (format {FORMAT}, step {config.STEP_DEG})")
            allowed = tuple(p + "/" for p in manifest["parts"])
            members = [m for m in z.namelist() if m != MANIFEST]
            bad = [m for m in members if not m.startswith(allowed) or ".." in m.split("/") or m.startswith("/")]
            if bad:
                raise SystemExit(f"refusing to install: unexpected paths in tileset, e.g. {bad[0]}")
            os.makedirs(config.DATA_DIR, exist_ok=True)
            z.extractall(config.DATA_DIR, members)
        st = status()
        print(f"Installed {len(members)} tiles into {config.DATA_DIR}")
        print(f"  offline ready: {st['offline_ready']}; cloud dates: {', '.join(st['cloud_dates']) or 'none'}")
        return st
    finally:
        if tmp:
            os.remove(tmp)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m phonehome.tileset", description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", help=f"tile cache (default: {config.DATA_DIR})")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="show what's in the local tile cache")
    pk = sub.add_parser("pack", help="zip the tile cache for offline use")
    pk.add_argument("out", help="output .zip path")
    pk.add_argument("--clouds", nargs="*", default=[], metavar="YYYY-MM-DD",
                    help="cloud dates to include (~45 MB each)")
    pk.add_argument("--all-clouds", action="store_true", help="include every cached cloud date")
    ins = sub.add_parser("install", help="install a tileset zip into the tile cache")
    ins.add_argument("src", nargs="?", help=f"zip path or URL (default: {config.TILESET_URL})")
    args = p.parse_args(argv)

    if args.data_dir:
        config.set_data_dir(args.data_dir)
    if args.cmd == "status":
        print(json.dumps(status(), indent=2))
    elif args.cmd == "pack":
        pack(args.out, cached_cloud_dates() if args.all_clouds else args.clouds)
    elif args.cmd == "install":
        install(args.src)


if __name__ == "__main__":
    sys.exit(main())
