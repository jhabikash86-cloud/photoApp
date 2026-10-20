"""Crop, straighten and aspect ratio: the final geometric step of an edit, one cubic warp from the finished picture.

A window of the chosen aspect ratio is placed inside the (rotated) picture so that it never shows an empty corner: the window is the largest
one that fits, optionally zoomed in and moved. Everything is stored as normalised numbers (angle, aspect, zoom, centre), so a preview and the
full-resolution export crop exactly the same area.

auto_angle() finds how much to rotate to level the picture: from the horizon (the lower edge of the sky or the top of the water) when there is
a clean one, otherwise from the dominant near-horizontal / near-vertical lines (buildings, fences, poles).
"""
import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from params import Geometry

ASPECTS: Dict[str, Optional[float]] = {
    "original": None, "free": None, "1:1": 1.0, "4:5": 4 / 5, "5:4": 5 / 4, "3:2": 3 / 2, "2:3": 2 / 3, "16:9": 16 / 9, "9:16": 9 / 16, "3:1": 3.0,
}


def active(g: Geometry) -> bool:
    return bool(g.enabled and (abs(g.angle) > 1e-3 or g.aspect not in ("original",) or g.zoom > 1.001))


def _fit(w: float, h: float, theta: float, aspect: float) -> Tuple[float, float]:
    """Largest (cw, ch) with cw/ch = aspect that fits inside a w x h rectangle rotated by theta."""
    t = abs(theta)
    c, s = math.cos(t), math.sin(t)
    ch = min(w / (aspect * c + s), h / (aspect * s + c))
    return aspect * ch, ch


def _inside(cw: float, ch: float, dx: float, dy: float, theta: float, w: float, h: float) -> bool:
    """Is the window (axis-aligned in the OUTPUT frame, centre moved by dx, dy) fully inside the source rectangle?"""
    c, s = math.cos(theta), math.sin(theta)
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            px, py = dx + sx * cw, dy + sy * ch
            u, v = c * px + s * py, -s * px + c * py             # back into the source frame (rotate by -theta)
            if abs(u) > w / 2 - 1.0 or abs(v) > h / 2 - 1.0:                  # one pixel of safety: the interpolation reads its neighbours
                return False
    return True


def _max_shift(cw, ch, theta, w, h, axis: int, sign: float) -> float:
    lo, hi = 0.0, max(w, h)
    for _ in range(24):
        mid = (lo + hi) / 2
        d = (sign * mid, 0.0) if axis == 0 else (0.0, sign * mid)
        if _inside(cw, ch, d[0], d[1], theta, w, h):
            lo = mid
        else:
            hi = mid
    return lo


def window(w: int, h: int, g: Geometry) -> Tuple[int, int, np.ndarray]:
    """(out width, out height, forward 2x3 matrix source -> output) for a w x h picture."""
    theta = math.radians(g.angle)
    a = ASPECTS.get(g.aspect)
    if g.aspect == "original":
        a = w / h
    if g.aspect == "free":
        a = w / h
    cw, ch = _fit(w, h, theta, a)
    if abs(theta) > 1e-6:
        k = 1.0 - 3.0 / max(w, h)                                      # a few pixels of safety so interpolation never reads outside
        cw, ch = cw * k, ch * k
    z = max(1.0, g.zoom)
    cw, ch = cw / z, ch / z
    # position: g.x, g.y in -1..1 move the window inside the room that is left
    dx = g.x * (_max_shift(cw, ch, theta, w, h, 0, 1.0) if g.x >= 0 else _max_shift(cw, ch, theta, w, h, 0, -1.0))
    dy = g.y * (_max_shift(cw, ch, theta, w, h, 1, 1.0) if g.y >= 0 else _max_shift(cw, ch, theta, w, h, 1, -1.0))
    if not _inside(cw, ch, dx, dy, theta, w, h):                       # both axes at their own limit can still leave a corner: pull back together
        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = (lo + hi) / 2
            if _inside(cw, ch, dx * mid, dy * mid, theta, w, h):
                lo = mid
            else:
                hi = mid
        dx, dy = dx * lo, dy * lo
    ow, oh = max(2, int(round(cw))), max(2, int(round(ch)))
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], np.float64)
    ctr_src = np.array([w / 2.0, h / 2.0])
    off = np.array([dx, dy])
    t = np.array([ow / 2.0, oh / 2.0]) - R @ ctr_src - off
    M = np.hstack([R, t[:, None]])
    return ow, oh, M


def apply(arr: np.ndarray, g: Geometry) -> np.ndarray:
    """The picture rotated and cropped. Returns `arr` itself when the geometry does nothing."""
    if not active(g):
        return arr
    h, w = arr.shape[:2]
    ow, oh, M = window(w, h, g)
    return cv2.warpAffine(arr, M, (ow, oh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


# ------------------------------------------------------------------------------------------------ auto straighten
def _fit_line(xs: np.ndarray, ys: np.ndarray, iters: int = 120):
    """Robust line y = a x + b (RANSAC); returns (slope, inlier fraction, residual in px of the inliers)."""
    rng = np.random.default_rng(3)
    n = len(xs)
    if n < 20:
        return 0.0, 0.0, 99.0
    best = (0.0, 0.0, 99.0)
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        if abs(xs[i] - xs[j]) < n * 0.15:
            continue
        a = (ys[j] - ys[i]) / (xs[j] - xs[i])
        b = ys[i] - a * xs[i]
        res = np.abs(ys - (a * xs + b))
        inl = res < 2.5
        f = float(inl.mean())
        if f > best[1]:
            a2, b2 = np.polyfit(xs[inl], ys[inl], 1) if inl.sum() > 10 else (a, b)
            best = (float(a2), f, float(np.abs(ys[inl] - (a2 * xs[inl] + b2)).mean()))
    return best


def _trend(xs: np.ndarray, ys: np.ndarray):
    """The overall tilt of a bumpy skyline (hills, trees): Theil-Sen slope of the median-smoothed edge. Returns (angle in degrees, spread in degrees);
    the spread is the interquartile range of all pairwise slopes, small when the line is really straight."""
    if len(xs) < 40:
        return 0.0, 99.0
    sm = cv2.medianBlur(ys.astype(np.float32).reshape(1, -1), 5).ravel().astype(np.float64)
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(xs), 6000), rng.integers(0, len(xs), 6000)
    far = np.abs(xs[i] - xs[j]) > len(xs) * 0.25
    if far.sum() < 200:
        return 0.0, 99.0
    ang = np.degrees(np.arctan((sm[j] - sm[i])[far] / (xs[j] - xs[i])[far]))
    q1, med, q3 = np.percentile(ang, [25, 50, 75])
    return float(-med), float(q3 - q1)


def auto_angle(rgb: np.ndarray, masks: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, object]:
    """How many degrees to rotate (positive = clockwise) to level the picture, with a confidence and where the estimate came from.
    rgb: float 0..1 (any size, it is looked at small)."""
    h0, w0 = rgb.shape[:2]
    s = min(1.0, 900 / max(h0, w0))
    img = cv2.resize(rgb, (max(8, int(w0 * s)), max(8, int(h0 * s))), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    h, w = img.shape[:2]
    out = {"angle": 0.0, "confidence": 0.0, "source": "none"}
    # 1. the horizon: the lower edge of the sky, or the top edge of water, as a straight line
    for name, lower in (("sky", True), ("water", False)):
        m = None if not masks or name not in masks else cv2.resize(np.asarray(masks[name], np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5
        if m is None or m.mean() < 0.06 or m.mean() > 0.85:
            continue
        cols = m.any(0)
        if cols.mean() < 0.6:
            continue
        idx = np.arange(h)[:, None]
        edge = np.where(cols, (np.where(m, idx, -1).max(0) if lower else np.where(m, idx, h + 1).min(0)), -1).astype(np.float64)
        xs = np.nonzero(cols)[0].astype(np.float64)
        ys = edge[cols]
        slope, frac, resid = _fit_line(xs, ys)
        ang = -math.degrees(math.atan(slope))
        if frac > 0.55 and abs(ang) < 10:
            conf = float(np.clip((frac - 0.5) * 2.0, 0, 1) * np.clip(1.4 - resid, 0, 1))
            if conf > out["confidence"]:
                out = {"angle": round(ang, 2), "confidence": round(conf, 2), "source": f"horizon ({name})"}
        else:                                                           # a bumpy skyline: use its overall trend
            ang, spread = _trend(xs, ys)
            conf = float(np.clip(1.0 - spread / 5.0, 0, 1)) * 0.9
            if abs(ang) < 10 and conf > out["confidence"]:
                out = {"angle": round(ang, 2), "confidence": round(conf, 2), "source": f"skyline trend ({name})"}
    if out["confidence"] >= 0.55:
        return out
    # 2. strong straight lines that should be horizontal or vertical
    gray = (cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 1.2), 60, 150)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 360, threshold=70, minLineLength=int(0.22 * min(h, w)), maxLineGap=6)
    if lines is not None:
        dev, wt = [], []
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            a = math.degrees(math.atan2(y2 - y1, x2 - x1))
            for base in (0.0, 90.0, -90.0, 180.0, -180.0):
                d = a - base
                if abs(d) < 8.0:
                    dev.append(d)
                    wt.append(math.hypot(x2 - x1, y2 - y1))
                    break
        if len(dev) >= 4:
            dev, wt = np.array(dev), np.array(wt)
            order = np.argsort(dev)
            cw = np.cumsum(wt[order]) / wt.sum()
            med = float(dev[order][np.searchsorted(cw, 0.5)])
            spread = float(np.sqrt(np.average((dev - med) ** 2, weights=wt)))
            conf = float(np.clip(1.0 - spread / 3.0, 0, 1) * np.clip(len(dev) / 25.0, 0, 1))
            ang = -med
            if conf > out["confidence"] and abs(ang) < 8:
                out = {"angle": round(ang, 2), "confidence": round(conf, 2), "source": "straight lines"}
    return out
