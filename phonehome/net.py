"""HTTP fetching with retries, and NASA GIBS WMS tile requests."""

import ssl
import time
import urllib.parse
import urllib.request

from . import config

# One SSL context built up front and shared: building it inside each worker thread can fail
# on Windows (sporadic FileNotFoundError during the handshake).
_ssl_ctx = None

# Set by --offline (or by the control panel when a tileset is installed and the user asks for
# offline): any download attempt then fails with a clear message instead of touching the network.
OFFLINE = False


class OfflineError(RuntimeError):
    pass


def init():
    """Call once on the main thread before any threaded downloads."""
    global _ssl_ctx
    if _ssl_ctx is None:
        _ssl_ctx = ssl.create_default_context()


def fetch(url, tries=4, timeout=60):
    """GET `url`, retrying transient network errors. Returns (content_type, bytes)."""
    if OFFLINE:
        raise OfflineError(f"offline mode: not downloading {url.split('?')[0]} -- data missing from the "
                           f"local cache; install the tileset (python -m phonehome.tileset install ...)")
    init()
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=_ssl_ctx) as r:
                return r.headers.get("Content-Type", ""), r.read()
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def gibs_tile(layer, lat0, lon0, px, date=None):
    """JPEG bytes of one STEP_DEG x STEP_DEG GIBS tile."""
    step = config.STEP_DEG
    params = {
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0",
        "LAYERS": layer, "STYLES": "", "CRS": "EPSG:4326",
        # WMS 1.3.0 + EPSG:4326 uses lat,lon axis order
        "BBOX": f"{lat0},{lon0},{lat0 + step},{lon0 + step}",
        "WIDTH": px, "HEIGHT": px, "FORMAT": "image/jpeg",
    }
    if date:
        params["TIME"] = date.isoformat()
    ctype, data = fetch(f"{config.GIBS_WMS}?{urllib.parse.urlencode(params)}")
    if not ctype.startswith("image/"):
        raise RuntimeError(f"GIBS returned {ctype} for {layer} {lat0},{lon0}: {data[:200]!r}")
    return data
