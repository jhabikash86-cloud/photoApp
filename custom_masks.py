"""User-created masks: linear gradient, radial, brush, luminosity range and colour range.

Every builder returns a float32 mask in 0..1 at the render size (w, h), described in picture-relative coordinates, so the
same mask is exact at preview size, zoomed size and full-resolution export. Shared finishing, in this order:
  edge-aware refine ("auto mask": the edge snaps to the picture's real edges, never bleeds across them)
  -> optional limit to (or exclusion of) an AI region such as the sky or the person
  -> invert -> strength.
Geometry masks (gradient, radial, brush) carry their own softness; range masks (luminosity, colour) use a spatial feather.
"""
from typing import Optional

import cv2
import numpy as np

from adjust import edge_aware_blur
from params import MaskLayer, Stroke

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _smooth(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------------------------------------ geometry
def linear(size, line, feather: float) -> np.ndarray:
    """Graduated filter. Full effect at (x1,y1), none at (x2,y2), a smooth ramp between, flat beyond both ends."""
    w, h = size
    x1, y1, x2, y2 = (float(v) for v in line)
    gx, gy = np.meshgrid((np.arange(w, dtype=np.float32) + 0.5) / w, (np.arange(h, dtype=np.float32) + 0.5) / h)
    # work in pixel units so the ramp is not skewed by the picture's aspect ratio
    dx, dy = (x2 - x1) * w, (y2 - y1) * h
    length = max(float(np.hypot(dx, dy)), 1e-3)
    t = ((gx - x1) * w * dx + (gy - y1) * h * dy) / (length * length)
    ramp = 1.0 - np.clip(t, 0, 1)                    # 1 at the start line, 0 at the end line
    width = 0.06 + 0.94 * np.clip(feather / 100.0, 0.0, 1.0)     # feather 0 = a tight edge in the middle, 100 = the whole span
    return _smooth((ramp - (0.5 - width / 2)) / width).astype(np.float32)


def radial(size, ell, feather: float) -> np.ndarray:
    """Elliptical filter: 1 in the middle, fading out through the edge. (invert it to darken or lighten the surroundings)"""
    w, h = size
    cx, cy, rx, ry, ang = (float(v) for v in ell)
    gx, gy = np.meshgrid((np.arange(w, dtype=np.float32) + 0.5), (np.arange(h, dtype=np.float32) + 0.5))
    px, py = gx - cx * w, gy - cy * h
    a = np.radians(ang)
    u = px * np.cos(a) + py * np.sin(a)
    v = -px * np.sin(a) + py * np.cos(a)
    d = np.sqrt((u / max(rx * w, 1.0)) ** 2 + (v / max(ry * h, 1.0)) ** 2)    # 1.0 on the ellipse
    soft = 0.02 + 0.98 * np.clip(feather / 100.0, 0, 1)
    return (1.0 - _smooth((d - (1.0 - soft)) / (2.0 * soft))).astype(np.float32)


def brush(size, strokes) -> np.ndarray:
    """Paint strokes. Each stroke is a soft round brush dragged along its points; erase strokes remove paint."""
    w, h = size
    edge = max(w, h)
    s = min(1.0, 900 / edge)                                  # strokes are soft, so rasterise at reduced size
    sw, sh = max(8, int(round(w * s))), max(8, int(round(h * s)))
    acc = np.zeros((sh, sw), np.float32)
    for st in strokes:
        if not st.pts:
            continue
        pts = np.array(st.pts, np.float32)
        pts = pts[~np.isnan(pts).any(axis=1)]
        if len(pts) == 0:
            continue
        r = (0.004 + 0.12 * st.size / 100.0) * max(sw, sh)       # brush radius in pixels
        layer = np.zeros((sh, sw), np.float32)
        xy = np.stack([pts[:, 0] * sw, pts[:, 1] * sh], axis=1)
        thickness = max(1, int(round(2 * r * (1.0 - 0.45 * st.feather / 100.0))))   # solid core; the feather adds the rest
        if len(xy) == 1:
            cv2.circle(layer, tuple(int(v) for v in xy[0]), max(1, thickness // 2), 1.0, -1, cv2.LINE_AA)
        else:
            cv2.polylines(layer, [np.round(xy).astype(np.int32).reshape(-1, 1, 2)], False, 1.0, thickness, cv2.LINE_AA)
            for p0 in (xy[0], xy[-1]):
                cv2.circle(layer, tuple(int(v) for v in p0), max(1, thickness // 2), 1.0, -1, cv2.LINE_AA)
        sigma = max(0.6, r * 0.9 * st.feather / 100.0)
        layer = np.clip(cv2.GaussianBlur(layer, (0, 0), sigma) * (1.0 + 1.4 * st.feather / 100.0), 0, 1) * (st.flow / 100.0)
        acc = acc * (1 - layer) if st.erase else np.maximum(acc, layer)
    if (sw, sh) != (w, h):
        acc = cv2.resize(acc, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(acc, 0, 1).astype(np.float32)


# ------------------------------------------------------------------------------------------------ range masks
def luminosity(src: np.ndarray, lo: float, hi: float, soft: float) -> np.ndarray:
    """Select by brightness (0..100). The brightness is read from an edge-aware blur so noise and texture do not speckle it."""
    y = np.clip(edge_aware_blur((src @ LUMA).astype(np.float32), max(src.shape[:2]) / 220, 0.02), 0, 1) * 100.0
    lo, hi = min(lo, hi), max(lo, hi)
    sf = 1.0 + 0.5 * soft
    up = _smooth((y - (lo - sf)) / (2 * sf))
    down = 1.0 - _smooth((y - (hi - sf)) / (2 * sf)) if hi < 99.5 else np.ones_like(y)
    up = up if lo > 0.5 else np.ones_like(y)
    return (up * down).astype(np.float32)


def colour(src: np.ndarray, pick, tol: float) -> np.ndarray:
    """Select everything whose colour is close to the picked pixel's colour (CIE Lab distance)."""
    h, w = src.shape[:2]
    px, py = int(np.clip(pick[0], 0, 1) * (w - 1)), int(np.clip(pick[1], 0, 1) * (h - 1))
    r = max(2, max(h, w) // 250)
    patch = src[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1].reshape(-1, 3)
    lab = cv2.cvtColor(np.clip(src, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    tgt = cv2.cvtColor(patch.mean(axis=0).reshape(1, 1, 3).astype(np.float32), cv2.COLOR_RGB2Lab)[0, 0]
    chroma = float(np.hypot(tgt[1], tgt[2]))
    dl = (lab[..., 0] - tgt[0]) * (0.35 if chroma > 12 else 1.0)       # for coloured picks, brightness matters less
    da, db = lab[..., 1] - tgt[1], lab[..., 2] - tgt[2]
    dist = np.sqrt(dl * dl + da * da + db * db)
    t = 6.0 + 0.6 * tol
    m = 1.0 - _smooth((dist - 0.45 * t) / (0.55 * t))
    # smooth out single-pixel colour noise without crossing edges
    return np.clip(edge_aware_blur(m.astype(np.float32), max(h, w) / 300, 0.04), 0, 1)


# ------------------------------------------------------------------------------------------------ assembly
def _guide(src: np.ndarray, m: np.ndarray, refine: float) -> np.ndarray:
    """Edge-aware snap ("auto mask"): the mask follows the picture's real edges instead of ignoring them."""
    from landscape import _guided
    h, w = m.shape
    s = min(1.0, 900 / max(h, w))
    g = cv2.cvtColor(np.clip(src, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    gs = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    ms = cv2.resize(m, (gs.shape[1], gs.shape[0]), interpolation=cv2.INTER_AREA) if s < 1 else m
    r = max(2, int(max(gs.shape) / (200 - 130 * refine / 100)))
    out = np.clip(_guided(gs, ms.astype(np.float32), r=r, eps=2e-3 * (1.0 - 0.9 * refine / 100)), 0, 1)
    if s < 1:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(m * (1 - refine / 100) + out * (refine / 100), 0, 1)


def build(layer: MaskLayer, size, src: np.ndarray, bank=None) -> Optional[np.ndarray]:
    """The mask for `layer` at `size` = (w, h). `src` is the picture at the same size (for range masks and edge snapping)."""
    k = layer.kind
    if k == "linear":
        m = linear(size, layer.line, layer.feather)
    elif k == "radial":
        m = radial(size, layer.ellipse, layer.feather)
    elif k == "brush":
        m = brush(size, layer.strokes)
    elif k == "luminosity":
        m = luminosity(src, layer.lum[0], layer.lum[1], layer.lum_soft)
    elif k == "color":
        m = colour(src, layer.pick, layer.tolerance)
    else:
        return None
    if k in ("luminosity", "color") and layer.feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), max(0.5, max(size) * 0.0012 + max(size) * 0.012 * layer.feather / 100))
    if layer.refine > 0:
        m = _guide(src, m, layer.refine)
    if layer.limit and bank is not None:
        lim = bank.get(layer.limit, size)
        if lim is not None:
            m = m * ((1.0 - lim) if layer.limit_invert else lim)
    if layer.invert:
        m = 1.0 - m
    return np.clip(m * (layer.strength / 100.0), 0, 1).astype(np.float32)
