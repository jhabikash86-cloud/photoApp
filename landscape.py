"""Landscape tools: dehaze, sky grading, targeted HSL, ground texture, split toning.

Arrays are float32 RGB in [0, 1]; functions return new arrays unless noted.
"""
from typing import Optional

import cv2
import numpy as np

from adjust import edge_aware_blur

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _resize(a: np.ndarray, size) -> np.ndarray:
    return cv2.resize(a, size, interpolation=cv2.INTER_CUBIC)


def _guided(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Edge-aware smoothing of `src` following `guide` (He et al. guided filter)."""
    box = lambda a: cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1))
    mg, ms = box(guide), box(src)
    cov = box(guide * src) - mg * ms
    var = box(guide * guide) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return box(a) * guide + box(b)


def dehaze_fields(arr: np.ndarray) -> tuple:
    """Dark-channel-prior haze model: (transmission map at full size, atmospheric light colour)."""
    H, W = arr.shape[:2]
    scale = min(1.0, 700 / max(H, W))
    small = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else arr
    k = max(3, int(max(small.shape[:2]) / 45)) | 1
    kernel = np.ones((k, k), np.uint8)
    dark = cv2.erode(small.min(axis=2), kernel)
    n = max(10, int(dark.size * 0.002))          # atmospheric light: haziest 0.2% of pixels
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    A = np.clip(small.reshape(-1, 3)[idx].mean(axis=0), 0.4, 0.97).astype(np.float32)
    t = 1 - 0.95 * cv2.erode((small / A).min(axis=2), kernel)
    t = _guided(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), t.astype(np.float32), r=max(8, k * 2), eps=1e-3)
    t = np.clip(t, 0.25, 1.0)
    if scale < 1:
        t = _resize(t, (W, H))
    return np.clip(t, 0.25, 1.0).astype(np.float32), A


def dehaze_apply(arr: np.ndarray, fields: tuple, amount: float, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """amount: -1 (add haze) .. +1 (remove haze), optionally only where mask > 0."""
    t, A = fields
    if abs(amount) < 1e-3:
        return arr
    if amount < 0:   # add atmosphere
        k = min(1.0, -amount) * 0.55
        out = arr * (1 - k) + A * k
        return out if mask is None else arr + (out - arr) * mask[..., None]
    tt = 1 - (1 - t) * min(1.5, amount)
    if mask is not None:
        tt = 1 - (1 - tt) * mask
    out = np.clip((arr - A) / np.maximum(tt, 0.2)[..., None] + A, 0, 1)
    # Dehaze darkens; restore part of the lost brightness so the image doesn't sink.
    sel = None if mask is None else mask > 0.5
    ref = np.median(arr) if sel is None or not sel.any() else np.median(arr[sel])
    res = np.median(out) if sel is None or not sel.any() else np.median(out[sel])
    ratio = (ref + 1e-3) / (res + 1e-3)
    gain = float(np.clip(ratio, 1.0, 1.7)) ** 0.85
    if mask is not None:
        gain = 1 + (gain - 1) * mask[..., None]
    return np.clip(out * gain, 0, 1)


def dehaze(arr: np.ndarray, strength: float, sky: Optional[np.ndarray] = None) -> np.ndarray:
    """Convenience wrapper: dehaze everything except the sky."""
    if strength <= 0.01:
        return arr
    mask = None if sky is None else 1 - 0.9 * np.clip(sky, 0, 1)
    return dehaze_apply(arr, dehaze_fields(arr), strength, mask)


def _hue_weight(hue: np.ndarray, center: float, width: float) -> np.ndarray:
    d = np.abs((hue - center + 180) % 360 - 180)
    return np.clip(1 - d / width, 0, 1) ** 1.5


def hsl_adjust(arr: np.ndarray, bands, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Hue-selective colour work. bands: (center_deg, width_deg, sat_gain, val_gain, hue_shift_deg)."""
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    ds = np.zeros_like(s)
    dv = np.zeros_like(v)
    dh = np.zeros_like(h)
    for center, width, sg, vg, hs in bands:
        w = _hue_weight(h, center, width) * np.clip(s * 2.5, 0, 1)  # skip near-neutral pixels
        ds += w * (sg - 1)
        dv += w * (vg - 1)
        dh += w * hs
    if mask is not None:
        ds, dv, dh = ds * mask, dv * mask, dh * mask
    hsv[..., 0] = (h + dh) % 360
    hsv[..., 1] = np.clip(s * (1 + ds), 0, 1)
    hsv[..., 2] = np.clip(v * (1 + dv), 0, 1)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1)


def local_contrast(arr: np.ndarray, amount: float, radius: float, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Midtone-weighted local contrast ("clarity"/"texture" depending on radius)."""
    y = arr @ LUMA
    detail = y - edge_aware_blur(y.astype(np.float32), radius, eps=0.05)  # edge-aware: no halos
    gain = amount * detail * 4 * y * (1 - y)
    if mask is not None:
        gain = gain * mask
    return np.clip(arr + gain[..., None], 0, 1)


def grade_sky(arr: np.ndarray, sky: np.ndarray, strength: float) -> np.ndarray:
    """Make the sky richer and add depth: graduated darkening, cloud contrast, colour."""
    H, W = arr.shape[:2]
    s = float(np.clip(strength, 0, 1))
    m = np.clip((np.clip(sky, 0, 1) - 0.35) / 0.65, 0, 1) ** 1.5  # tighten edges, avoid halos
    rows = m.mean(axis=1)
    bottom = int(np.max(np.nonzero(rows > 0.2)[0])) + 1 if (rows > 0.2).any() else int(H * 0.4)
    grad = np.clip(1 - np.arange(H, dtype=np.float32) / max(bottom, 1), 0, 1) ** 0.8
    grad = grad[:, None]

    sky_px = m > 0.5
    hsv_mean = cv2.cvtColor(arr[sky_px].reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3).mean(axis=0)
    bluish = 185 < hsv_mean[0] < 255 and hsv_mean[1] > 0.12

    out = arr.copy()
    # A blown-out sky has no detail to recover; darkening it only makes flat grey.
    blown = float(np.clip((float((arr @ LUMA)[sky_px].mean()) - 0.82) / 0.12, 0, 1))
    overcast = float(np.clip((0.22 - float(hsv_mean[1])) / 0.15, 0, 1))  # pale/grey sky: darkening only dulls it
    s_tone = s * (1 - blown) * (1 - 0.6 * overcast)
    # Highlight compression + graduated darkening toward the top of the frame.
    y = out @ LUMA
    compress = 1 - 0.20 * s_tone * np.clip((y - 0.55) / 0.45, 0, 1)
    dark = 1 - 0.20 * s_tone * grad
    out *= (1 + (compress * dark - 1) * m)[..., None]
    # Cloud structure: large-radius local contrast confined to the sky.
    k = max(3, int(max(H, W) / 120)) | 1  # keep contrast work away from object edges (halos)
    inner = cv2.GaussianBlur(cv2.erode(m, np.ones((k, k), np.uint8)), (0, 0), k / 2)
    out = local_contrast(out, 0.9 * s_tone, max(H, W) / 40, inner)
    if bluish:
        out = hsl_adjust(out, [(215, 45, 1 + 0.55 * s, 0.93, -4)], m)
    else:
        out = hsl_adjust(out, [(25, 40, 1 + 0.08 * s, 1.0, 0), (350, 40, 1 + 0.06 * s, 1.0, 0)], m)
    return np.clip(out, 0, 1)


def split_tone(arr: np.ndarray, strength: float, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Warm highlights, cool shadows: the classic landscape grade, kept subtle."""
    y = (arr @ LUMA)[..., None]
    hi = np.clip((y - 0.5) / 0.5, 0, 1)
    lo = np.clip((0.5 - y) / 0.5, 0, 1)
    warm = np.array([0.030, 0.010, -0.025], np.float32) * strength
    cool = np.array([-0.012, 0.004, 0.028], np.float32) * strength
    tone = hi * warm + lo * cool
    if mask is not None:  # never tint the sky: warm highlights turn white clouds beige
        tone = tone * mask[..., None]
    return np.clip(arr + tone, 0, 1)
