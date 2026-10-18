"""Monocular depth (Depth Anything V2 Small, ONNX, runs locally). Used for real, depth-based bokeh.

Returns relative disparity in 0..1 where 1 = nearest to the camera, edge-snapped to the photo.
"""
import json
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from landscape import _guided

MODEL = Path(__file__).parent / "models" / "depth_anything_v2_small.onnx"
LONG_EDGE = 756          # multiple of 14, as the network requires
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_session = None
_lock = threading.Lock()


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return MODEL.exists()


def _load():
    global _session
    if _session is None:
        import onnxruntime as ort
        _session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])


def estimate(rgb: np.ndarray) -> Optional[np.ndarray]:
    """rgb float32 [0,1] -> float32 HxW disparity in 0..1 (1 = near), or None if the model is missing."""
    if not available():
        return None
    _load()
    h, w = rgb.shape[:2]
    s = LONG_EDGE / max(h, w)
    nw, nh = max(14, int(round(w * s / 14)) * 14), max(14, int(round(h * s / 14)) * 14)
    img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x = ((img - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    with _lock:
        d = _session.run(None, {"pixel_values": x})[0][0]
    lo, hi = np.percentile(d, [2, 98])
    d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    # snap depth edges to the photo's edges so blur boundaries follow real object boundaries
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sm = min(1.0, 900 / max(h, w))
    gs = cv2.resize(guide, None, fx=sm, fy=sm, interpolation=cv2.INTER_AREA)
    ds = cv2.resize(d, (gs.shape[1], gs.shape[0]), interpolation=cv2.INTER_AREA)
    ds = np.clip(_guided(gs, ds, r=max(4, max(gs.shape) // 120), eps=2e-3), 0, 1)
    return cv2.resize(ds, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1).astype(np.float32)
