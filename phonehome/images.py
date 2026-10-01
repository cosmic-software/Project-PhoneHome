"""Image read/write via Blender's bundled OpenImageIO."""

import os
import tempfile

import numpy as np
import OpenImageIO as oiio


def read(path, fmt=oiio.FLOAT):
    """(h, w, channels) array."""
    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise RuntimeError(f"could not open {path}: {oiio.geterror()}")
    px = np.asarray(inp.read_image(fmt))
    inp.close()
    return px


def read_rgb(path):
    return read(path)[:, :, :3]


def decode_rgb(data, suffix=".jpg"):
    """Decode image bytes (OIIO has no in-memory reader in its Python API)."""
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return read_rgb(tmp)
    finally:
        os.remove(tmp)


def write_gray8(path, arr):
    """Write a 2-D uint8 array as a single-channel image."""
    h, w = arr.shape
    out = oiio.ImageOutput.create(path)
    out.open(path, oiio.ImageSpec(w, h, 1, oiio.UINT8))
    # (h, w, channels) explicitly -- OIIO writes a bare 2-D array transposed.
    out.write_image(np.ascontiguousarray(arr[:, :, np.newaxis]))
    out.close()


def resize_nearest(img, n):
    h, w = img.shape[:2]
    if (h, w) == (n, n):
        return img
    return img[np.ix_(np.linspace(0, h - 1, n).astype(int), np.linspace(0, w - 1, n).astype(int))]
