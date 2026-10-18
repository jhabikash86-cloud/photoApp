"""Natural-colour analysis (white balance) and Photoshop-style colour balance.

Several independent estimators each look at a different cue for "what colour is the light":
skin tone, white/grey clouds, bright near-whites, neutral midtones (gray-world) and edge statistics
(gray-edge). They vote in log-chromaticity space; outliers are dropped. Warm golden-hour light is
treated as mood and is only partly neutralised. Gains act on linear RGB, green fixed at 1.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np
from bands import banded

GAMMA = 2.2
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
SKIN_HUE_TARGET = 52.0    # Lab hue angle (degrees) of natural skin, near-constant across skin tones
SKIN_HUE_TOL = 6.0
MAX_LOG_SHIFT = 0.22      # never move a channel more than ~25% in auto mode
BASE_STRENGTH = 0.85      # auto correction is not full neutralisation


def temp_tint_gains(temp: float, tint: float) -> np.ndarray:
    """Slider -> linear RGB gains. +temp warmer, +tint magenta. Range -100..100."""
    return np.array([2 ** (temp / 100 * 0.5), 2 ** (-tint / 100 * 0.3), 2 ** (-temp / 100 * 0.5)], np.float32)


def _norm(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, np.float32)
    g = g / g[1]
    return g / float(np.dot(g, LUMA))  # keep overall brightness


def _from_neutral(c: np.ndarray) -> np.ndarray:
    c = np.maximum(c, 1e-4)
    return np.array([c[1] / c[0], 1.0, c[1] / c[2]], np.float32)


def _log2d(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, np.float32) / g[1]
    return np.log(np.array([g[0], g[2]]))


def _from_log2d(v: np.ndarray) -> np.ndarray:
    return np.array([np.exp(v[0]), 1.0, np.exp(v[1])], np.float32)


@dataclass
class WBResult:
    gains: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))
    confidence: float = 0.0
    estimates: Dict[str, dict] = field(default_factory=dict)
    candidates: Dict[str, np.ndarray] = field(default_factory=dict)
    warm_mood: bool = False
    note: str = ""

    def summary(self) -> dict:
        kelvin_shift = float(np.log(self.gains[0] / self.gains[2]))
        return {"confidence": round(self.confidence, 2), "warm_mood": self.warm_mood, "note": self.note,
                "shift": "warmer" if kelvin_shift > 0.02 else "cooler" if kelvin_shift < -0.02 else "none",
                "estimates": {k: {"weight": round(v["weight"], 2), "used": v["used"]} for k, v in self.estimates.items()},
                "candidates": list(self.candidates)}


def _skin_estimate(lin_px: np.ndarray) -> Optional[np.ndarray]:
    """Search temp/tint so the median skin lands in the natural skin-hue band."""
    med = np.median(lin_px, axis=0)
    best, best_cost = None, 1e9
    for t in range(-40, 41, 4):
        for ti in range(-20, 21, 4):
            g = temp_tint_gains(t, ti)
            c = np.clip(med * g, 0, 1) ** (1 / GAMMA)
            lab = cv2.cvtColor(c.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_RGB2Lab).reshape(3)
            hue = np.degrees(np.arctan2(lab[2], lab[1]))
            cost = max(0.0, abs(hue - SKIN_HUE_TARGET) - SKIN_HUE_TOL) + 0.02 * abs(t) + 0.03 * abs(ti)
            if cost < best_cost:
                best, best_cost = g, cost
    return best


def estimate(rgb: np.ndarray, sky: Optional[np.ndarray] = None, skin: Optional[np.ndarray] = None,
             foliage: Optional[np.ndarray] = None) -> WBResult:
    """rgb: float32 [0,1] gamma-encoded. Masks are HxW float (any size, resized here)."""
    s = min(1.0, 400 / max(rgb.shape[:2]))
    t = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
    h, w = t.shape[:2]

    def mk(m):
        return np.zeros((h, w), np.float32) if m is None else cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)

    sky_m, skin_m, fol_m = mk(sky), mk(skin), mk(foliage)
    lin = np.power(np.clip(t, 0, 1), GAMMA)
    y = t @ LUMA
    chroma = t.max(axis=2) - t.min(axis=2)

    est: Dict[str, dict] = {}

    # Neutral midtones, away from sky, foliage and skin (none of which are grey).
    pool = (sky_m < 0.3) & (fol_m < 0.3) & (skin_m < 0.3) & (y > 0.1) & (y < 0.9) & (chroma < 0.30)
    if pool.mean() > 0.01:
        est["gray_world"] = dict(g=_from_neutral(lin[pool].mean(axis=0)), weight=0.16, used=False)

    # Brightest unclipped near-whites outside the sky.
    cand = (sky_m < 0.3) & (t.max(axis=2) < 0.97) & (chroma < 0.28)
    if cand.sum() > 200:
        thr = np.percentile(y[cand], 99.3)
        top = cand & (y >= thr)
        if top.sum() >= 30 and thr > 0.45:
            est["white_patch"] = dict(g=_from_neutral(lin[top].mean(axis=0)), weight=0.12, used=False)

    # Grey/white clouds are an excellent neutral reference (blue sky is not).
    clouds = (sky_m > 0.6) & (y > 0.65) & (chroma < 0.16)
    if clouds.sum() > 120:
        est["clouds"] = dict(g=_from_neutral(lin[clouds].mean(axis=0)), weight=0.30, used=False)

    # Gray-edge: mean colour of derivatives is neutral under a uniform illuminant.
    blur = cv2.GaussianBlur(lin, (0, 0), 1.0)
    e = []
    for c in range(3):
        d = np.hypot(cv2.Sobel(blur[..., c], cv2.CV_32F, 1, 0), cv2.Sobel(blur[..., c], cv2.CV_32F, 0, 1))
        e.append(float((np.abs(d[(sky_m < 0.3)]) ** 5).mean() ** (1 / 5)) if (sky_m < 0.3).sum() > 500 else 0)
    if min(e) > 1e-4:
        est["gray_edge"] = dict(g=_from_neutral(np.array(e, np.float32)), weight=0.12, used=False)

    # Skin must sit in the natural skin-hue band.
    sk = skin_m > 0.7
    if sk.sum() > 300:
        g = _skin_estimate(lin[sk])
        if g is not None:
            est["skin"] = dict(g=g, weight=0.30, used=False)

    res = WBResult(estimates=est)
    if not est:
        res.note = "no reliable colour reference found; left as shot"
        res.candidates = {"asshot": np.ones(3, np.float32)}
        return res

    # Vote in log-chromaticity space, drop outliers.
    names = list(est)
    V = np.array([_log2d(est[n]["g"]) for n in names])
    W = np.array([est[n]["weight"] for n in names])
    med = np.array([np.median(V[:, 0]), np.median(V[:, 1])]) if len(names) > 2 else (V * W[:, None]).sum(0) / W.sum()
    dist = np.linalg.norm(V - med, axis=1)
    keep = dist < 0.18
    if keep.sum() < min(2, len(names)):
        keep = dist <= np.sort(dist)[min(1, len(names) - 1)]
    for n, k in zip(names, keep):
        est[n]["used"] = bool(k)
    v = (V[keep] * W[keep, None]).sum(0) / W[keep].sum()
    spread = float(dist[keep].mean())
    res.confidence = float(np.clip(1 - spread / 0.15, 0, 1) * min(1.0, W[keep].sum() / 0.4))

    # Golden hour / sunset: warm light is mood. Only partly neutralise a cooling correction.
    warm = False
    if sky is not None:
        hsv = cv2.cvtColor(t, cv2.COLOR_RGB2HSV)
        warm_px = (sky_m > 0.4) & (hsv[..., 1] > 0.3) & ((hsv[..., 0] < 55) | (hsv[..., 0] > 330))
        warm = bool(warm_px.sum() > 0.04 * h * w)
    if not warm:
        warm = bool((chroma[(y > 0.3)].mean() > 0.22) and (lin[..., 0].mean() > 1.35 * lin[..., 2].mean()))
    res.warm_mood = warm
    if warm and v[1] > v[0]:   # correction would cool the scene
        v = v * np.array([0.35, 0.35])
        res.note = "warm light preserved"

    # Trust scales with agreement between the estimators: unsure -> stay near as-shot.
    v = np.clip(v * BASE_STRENGTH * (0.35 + 0.65 * res.confidence), -MAX_LOG_SHIFT, MAX_LOG_SHIFT)
    res.gains = _norm(_from_log2d(v))

    full = _norm(_from_log2d(np.clip((V[keep] * W[keep, None]).sum(0) / W[keep].sum(), -0.3, 0.3)))
    res.candidates = {
        "auto": res.gains,
        "asshot": np.ones(3, np.float32),
        "neutral": full,
        "warm": _norm(res.gains * temp_tint_gains(14, 0)),
        "cool": _norm(res.gains * temp_tint_gains(-14, 0)),
    }
    if "skin" in est:
        res.candidates["skin"] = _norm(est["skin"]["g"])
    return res


@banded
def apply_gains(arr: np.ndarray, gains: np.ndarray, temp: float = 0.0, tint: float = 0.0) -> np.ndarray:
    """Apply white-balance gains (plus manual temp/tint) in linear light. Returns a new array."""
    g = np.asarray(gains, np.float32) * temp_tint_gains(temp, tint)
    if np.allclose(g, 1.0, atol=1e-3):
        return arr
    lin = np.power(np.clip(arr, 0, 1), GAMMA) * g
    return np.clip(np.power(np.clip(lin, 0, None), 1 / GAMMA), 0, 1)


@banded
def color_balance(arr: np.ndarray, shadows=(0, 0, 0), mids=(0, 0, 0), highs=(0, 0, 0)) -> np.ndarray:
    """Photoshop 'Color Balance'. Per tonal range: (cyan<->red, magenta<->green, yellow<->blue), -100..100.
    Positive = toward red / green / blue; the opposite colour is reduced so brightness stays put."""
    if not any(shadows) and not any(mids) and not any(highs):
        return arr
    y = (arr @ LUMA)[..., None]
    wt = {"s": (1 - y) ** 2, "m": 4 * y * (1 - y), "h": y ** 2}
    out = arr.copy()
    axes = (np.array([1.0, -0.5, -0.5], np.float32), np.array([-0.5, 1.0, -0.5], np.float32),
            np.array([-0.5, -0.5, 1.0], np.float32))
    for key, vals in (("s", shadows), ("m", mids), ("h", highs)):
        delta = sum(v / 100 * 0.10 * ax for v, ax in zip(vals, axes))
        if np.any(delta):
            out += wt[key] * delta
    return np.clip(out, 0, 1)
