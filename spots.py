"""Sensor dust in the sky: the faint, soft, round dark spots a dirty sensor leaves on a clear sky at small apertures.

They are found only INSIDE the sky, away from the skyline and from clouds, and only when they look like dust: round, soft-edged and faint. A bird, a plane or a
star is sharper, smaller or darker and is left alone. Each spot is replaced by the smooth sky around it (with matching grain), so nothing else in the picture changes.
"""
import math
from typing import Dict, List

import cv2
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
WORK = 1400


def detect(rgb: np.ndarray, sky: np.ndarray, amount: float = 60.0) -> List[dict]:
    """Dust spots as dicts (cx, cy, r in the picture's own pixels). amount 0..100: how faint a spot may be and still count."""
    H, W = rgb.shape[:2]
    s = min(1.0, WORK / max(H, W))
    sw, sh = max(8, int(W * s)), max(8, int(H * s))
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    y = (small @ LUMA).astype(np.float32)
    big = max(sw, sh)
    m = cv2.resize(np.asarray(sky, np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0.5
    if m.mean() < 0.04:
        return []
    re = max(5, int(0.02 * big))
    inner = cv2.erode(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * re + 1, 2 * re + 1))).astype(bool)
    if inner.sum() < 2000:
        return []
    k = int(max(31, 0.035 * big)) | 1
    bg = cv2.medianBlur((np.clip(y, 0, 1) * 255).astype(np.uint8), k).astype(np.float32) / 255.0
    diff = bg - y
    thr = 0.012 + (1.0 - np.clip(amount / 100.0, 0, 1)) * 0.030
    cand = ((diff > thr) & inner).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(cand)
    local_sd = cv2.GaussianBlur((y - cv2.GaussianBlur(y, (0, 0), 6)) ** 2, (0, 0), 12) ** 0.5       # how textured the sky is around here
    out = []
    amax = math.pi * (0.025 * big) ** 2
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < 5 or a > amax:
            continue
        bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if max(bw, bh) > 1.8 * max(1, min(bw, bh)):
            continue                                                   # a streak, a wire, a bird's wing: not dust
        if a / (math.pi * (max(bw, bh) / 2.0) ** 2) < 0.5:
            continue
        sel = lab == i
        contrast = float(diff[sel].mean())
        if contrast > 0.16:
            continue                                                   # too dark to be dust: a bird, a plane, a real object
        cx, cy = cen[i]
        if local_sd[int(cy), int(cx)] > 0.02:
            continue                                                   # cloud texture around it
        r = math.sqrt(a / math.pi)
        # soft edge: dust is out of focus, so its edge is a gentle slope; a sharp little speck has a steep one
        gy, gx = np.gradient(y)
        gmax = float(np.hypot(gx, gy)[cv2.dilate(sel.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0].max())
        if r < 3.5 and contrast > 0.05 and gmax > 0.9 * contrast:
            continue
        out.append(dict(cx=float(cx / s), cy=float(cy / s), r=float(r / s), contrast=round(contrast, 3)))
    return out


def clean(src: np.ndarray, masks: Dict[str, np.ndarray], amount: float = 60.0):
    """(cleaned picture, number of spots repaired). Returns (src, 0) when there is nothing to do."""
    if amount <= 0 or "sky" not in masks:
        return src, 0
    spots = detect(src, masks["sky"], amount)
    if not spots:
        return src, 0
    H, W = src.shape[:2]
    out = src.copy()
    rng = np.random.default_rng(1)
    for sp in spots:
        R = max(3.0, sp["r"] * 1.45 + 2.0)
        pad = int(R * 3) + 2
        x0, x1 = max(0, int(sp["cx"] - pad)), min(W, int(sp["cx"] + pad) + 1)
        y0, y1 = max(0, int(sp["cy"] - pad)), min(H, int(sp["cy"] + pad) + 1)
        roi = out[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.hypot(xx - sp["cx"], yy - sp["cy"])
        hole = (d <= R).astype(np.float32)
        ring = ((d > R * 1.5) & (d < R * 2.6)).astype(np.float32)
        if ring.sum() < 8:
            continue
        w = 1.0 - hole
        sig = max(2.0, R * 0.9)
        num = cv2.GaussianBlur(roi * w[..., None], (0, 0), sig)
        den = cv2.GaussianBlur(w, (0, 0), sig)[..., None]
        fill = num / np.maximum(den, 1e-3)
        res = roi - cv2.GaussianBlur(roi, (0, 0), 2.0)
        grain = float(res[ring > 0].std()) * 0.7
        fill = fill + rng.normal(0, grain, fill.shape).astype(np.float32)
        feather = cv2.GaussianBlur(hole, (0, 0), max(1.0, R * 0.30))[..., None]
        out[y0:y1, x0:x1] = np.clip(roi * (1 - feather) + fill * feather, 0, 1)
    return out, len(spots)
