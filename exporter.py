"""Export presets: the same finished picture delivered for the web, social, print or archive.

The edit is rendered once at full resolution in float, then (if the preset asks) reduced with area averaging, sharpened for its output size and
written in the right format: JPEG / WebP with the original EXIF and colour profile, or 16-bit PNG / TIFF for print and further editing.
"""
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import adjust
import engine
from params import EditParams

PRESETS: Dict[str, dict] = {
    "full": dict(label="Full size · JPEG (q95)", ext="jpg", format="jpeg", max_edge=None, quality=95, sharpen=0.0),
    "web": dict(label="Web · 2048 px JPEG", ext="jpg", format="jpeg", max_edge=2048, quality=88, sharpen=35.0),
    "social": dict(label="Social · 1350 px JPEG", ext="jpg", format="jpeg", max_edge=1350, quality=90, sharpen=40.0),
    "tiny": dict(label="Preview · 1024 px WebP", ext="webp", format="webp", max_edge=1024, quality=82, sharpen=40.0),
    "print": dict(label="Print master · 16-bit TIFF", ext="tif", format="tiff16", max_edge=None, quality=0, sharpen=0.0),
    "png16": dict(label="Archive · 16-bit PNG", ext="png", format="png16", max_edge=None, quality=0, sharpen=0.0),
}


def _unsharp16(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return arr
    sig = max(0.8, max(arr.shape[:2]) / 3000.0)
    blur = cv2.GaussianBlur(arr, (0, 0), sig)
    return np.clip(arr + (amount / 100.0) * 1.2 * (arr - blur), 0, 1)


def export(an, params: EditParams, preset: str, path_stem: str, max_edge: Optional[int] = None, quality: Optional[int] = None) -> Tuple[str, Tuple[int, int]]:
    """Render and write one picture; returns (file path, (width, height)). path_stem has no extension."""
    spec = dict(PRESETS.get(preset, PRESETS["full"]))
    if max_edge:
        spec["max_edge"] = max_edge
    if quality:
        spec["quality"] = quality
    arr = engine.render_float(an, params, None)
    long_edge = max(arr.shape[:2])
    me = spec["max_edge"]
    if me and long_edge > me:
        s = me / long_edge
        arr = cv2.resize(arr, (max(1, int(round(arr.shape[1] * s))), max(1, int(round(arr.shape[0] * s)))), interpolation=cv2.INTER_AREA)
    sharp = max(spec["sharpen"], params.effects.sharpen * min(1.0, params.intensity))
    path = f"{path_stem}.{spec['ext']}"
    if spec["format"] in ("tiff16", "png16"):
        a16 = (np.clip(_unsharp16(arr, sharp), 0, 1) * 65535.0 + 0.5).astype(np.uint16)[..., ::-1]     # BGR for OpenCV
        flags = [cv2.IMWRITE_TIFF_COMPRESSION, 5] if spec["format"] == "tiff16" else [cv2.IMWRITE_PNG_COMPRESSION, 4]
        if not cv2.imwrite(path, np.ascontiguousarray(a16), flags):
            raise RuntimeError("Could not write the file")
        return path, (arr.shape[1], arr.shape[0])
    img = adjust.sharpen_image(adjust.to_image(arr), sharp)
    original = adjust.load(an.path, 64)                         # only for the EXIF and the colour profile
    kw: dict = dict(quality=int(spec["quality"]))
    icc, exif = original.info.get("icc_profile"), original.info.get("exif")
    if icc:
        kw["icc_profile"] = icc
    if exif:
        kw["exif"] = exif
    if spec["format"] == "jpeg":
        img.save(path, "JPEG", subsampling=0, optimize=True, **kw)
    else:
        img.save(path, "WEBP", method=4, **kw)
    return path, img.size
