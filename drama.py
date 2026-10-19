"""The Drama stage: the part of a pro landscape edit that sliders alone cannot do.

An audit of professional before/after pairs found that one global curve explains at most half of such an edit. The missing tools are:

  MOOD      colour poured over ALL tones, greys included (Lab split-tone). Saturation / vibrance cannot colour a grey pixel; this can.
            Already-colourful pixels (foliage, flowers) and skin are protected, so greens stay lush and faces stay natural.
  SKY       a graduated darkening toward the top that keeps the cloud texture (a grey slab is what a plain exposure cut gives),
            plus extra colour in the sky.
  LIGHT     bloom around the brightest light, a luminous mist on the horizon, and atmospheric depth (distant hills sink into haze).

Everything works on a float32 RGB picture in 0..1 and is driven by the explicit `Drama` settings, so a look is a set of editable numbers.
"""
import math
from typing import Dict, Optional

import cv2
import numpy as np

from bands import banded
from params import Drama

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _ab(hue_deg: float, chroma: float) -> np.ndarray:
    h = math.radians(hue_deg)
    return np.array([chroma * math.cos(h), chroma * math.sin(h)], np.float32)


def _smooth_luma(arr: np.ndarray, frac: float = 80.0) -> np.ndarray:
    """Luminance blurred over about 1/80 of the picture: tonal-range membership that does not flicker with texture (cheap: done small)."""
    h, w = arr.shape[:2]
    s = min(1.0, 480 / max(h, w))
    small = cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else arr
    y = cv2.GaussianBlur((small @ LUMA).astype(np.float32), (0, 0), max(small.shape[:2]) / frac * 1.0)
    return cv2.resize(y, (w, h), interpolation=cv2.INTER_LINEAR) if s < 1 else y


# ------------------------------------------------------------------------------------------------ mood
@banded
def _colourise(arr: np.ndarray, ys: np.ndarray, protect_map: np.ndarray, targets: np.ndarray, strength: float, protect: float) -> np.ndarray:
    """Pull every pixel's colour toward the grade's colour for its tonal range, keeping its brightness.
    targets: (3, 2) Lab a,b for shadows / mids / highlights. protect_map: 0..1 per pixel (skin, extra sky strength folded into `strength` map)."""
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2Lab)
    w = np.stack([np.clip(1 - ys / 0.42, 0, 1), np.clip(1 - np.abs(ys - 0.5) / 0.32, 0, 1), np.clip((ys - 0.38) / 0.42, 0, 1)], -1)
    w /= w.sum(-1, keepdims=True) + 1e-6
    ta = w @ targets[:, 0]
    tb = w @ targets[:, 1]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    keep = np.clip(chroma / 38.0, 0, 1) * protect                       # colourful pixels keep their own colour
    st = np.clip(strength * (1.0 - keep) * protect_map, 0, 1)
    lab[..., 1] += (ta - lab[..., 1]) * st
    lab[..., 2] += (tb - lab[..., 2]) * st
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


@banded
def _enrich(arr: np.ndarray, amount: float, protect_map: np.ndarray) -> np.ndarray:
    """Richer existing colour: a chroma boost that tapers off for already-vivid pixels, so nothing clips to a flat colour."""
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2Lab)
    c = np.hypot(lab[..., 1], lab[..., 2])
    f = 1.0 + amount * np.clip(1.0 - c / 80.0, 0.0, 1.0) * protect_map
    lab[..., 1] *= f
    lab[..., 2] *= f
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


# ------------------------------------------------------------------------------------------------ sky
def _horizon(sky: Optional[np.ndarray]) -> Optional[float]:
    if sky is None or float((sky > 0.5).mean()) < 0.02:
        return None
    rows = np.nonzero(sky > 0.5)[0]
    return float(np.percentile(rows, 98)) / sky.shape[0]


@banded
def _sky_colour(arr: np.ndarray, ys: np.ndarray, m: np.ndarray, targets: np.ndarray, strength: float) -> np.ndarray:
    """Colour the sky without overwriting what it already has: a sky with colour of its own (blue day, sunset) is made RICHER in its own hue;
    only a grey or hazy sky is pulled toward the look's palette."""
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2Lab)
    w = np.stack([np.clip(1 - ys / 0.42, 0, 1), np.clip(1 - np.abs(ys - 0.5) / 0.32, 0, 1), np.clip((ys - 0.38) / 0.42, 0, 1)], -1)
    w /= w.sum(-1, keepdims=True) + 1e-6
    pa, pb = w @ targets[:, 0], w @ targets[:, 1]
    a, b = lab[..., 1], lab[..., 2]
    keep = np.clip(np.hypot(a, b) / 22.0, 0, 1)                         # 1 = the sky already has real colour
    ta = pa * (1 - keep) + a * (1.0 + 0.30 * strength) * keep
    tb = pb * (1 - keep) + b * (1.0 + 0.30 * strength) * keep
    st = np.clip(strength, 0, 1) * m
    lab[..., 1] = a + (ta - a) * st
    lab[..., 2] = b + (tb - b) * st
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def _level_horizon(sky: np.ndarray) -> bool:
    """True when the sky meets the land along a fairly level, unbroken line (sea or hills), not through branches and tree tops."""
    h, w = sky.shape
    cols = (sky > 0.5)
    has = cols.any(0)
    if has.mean() < 0.7:
        return False
    low = np.where(has, h - 1 - np.argmax(cols[::-1], 0), -1).astype(np.float32)[has] / h
    return float(np.percentile(low, 85) - np.percentile(low, 15)) < 0.15


def _sky(arr: np.ndarray, sky: np.ndarray, d: Drama, k: float, targets: np.ndarray) -> np.ndarray:
    """Graduated burn toward the top that keeps the clouds' own contrast, contrast inside real clouds only, and richer sky colour."""
    h, w = sky.shape
    hz = _horizon(sky)
    if hz is None:
        return arr
    sky0 = sky
    r = max(3, int(round(0.012 * max(h, w))))
    grown = cv2.dilate((sky > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(np.float32)
    sky = np.maximum(sky, cv2.GaussianBlur(grown, (0, 0), r / 2.0))      # the mask's soft edge must not leave an unburned bright rim along the skyline
    m = sky[..., None]
    out = arr
    burn = d.sky_burn / 100.0 * k
    if burn > 0:
        yy = np.arange(h, dtype=np.float32)[:, None] / (hz * h)
        ramp = np.clip(1.0 - yy, 0, 1) ** 1.15                         # 1 at the top of the frame, 0 at the horizon
        yl = _smooth_luma(out)
        room = np.clip((yl - 0.30) / 0.40, 0, 1)                        # a sky that is already dark or deep is not burned further
        f = 1.0 - 0.34 * burn * ramp * room                             # gentle: a pro lowers a bright sky by ~20%, it must stay luminous
        out = out * (1 + (f[..., None] - 1) * m)
    tex = d.sky_texture / 100.0 * k
    if tex > 0:
        y = (out @ LUMA).astype(np.float32)
        core = cv2.erode((sky0 > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(np.float32)
        core = cv2.GaussianBlur(core, (0, 0), r / 2.0)                          # texture only well inside the sky, never at the skyline
        sm = (sky0 > 0.5).astype(np.float32)

        def smooth(sig):                                                        # blur that only looks at SKY pixels (a dark hill must not read as a cloud edge)
            return cv2.GaussianBlur(y * sm, (0, 0), sig) / np.maximum(cv2.GaussianBlur(sm, (0, 0), sig), 1e-3)
        det = (y - smooth(max(2.0, h / 45))) * sm
        fine = (y - smooth(max(1.5, h / 150))) * sm
        soft = lambda x, t: np.sign(x) * np.maximum(np.abs(x) - t, 0.0)         # ignore gradient steps and noise: only real structure
        cloudy = cv2.GaussianBlur(np.abs(soft(det, 0.012)), (0, 0), max(4.0, h / 12))
        cloudy = np.clip((cloudy - 0.006) / 0.02, 0, 1)                         # 0 for a clear sky, 1 for a cloudy one
        boost = (0.30 * np.tanh(soft(det, 0.012) / 0.06) + 0.14 * np.tanh(soft(fine, 0.006) / 0.03)) * core
        out = np.clip(out + (tex * boost * cloudy)[..., None] * m, 0, 1)
    col = d.sky_colour / 100.0 * k
    if col > 0:
        ys = _smooth_luma(out)
        st = targets
        if d.sky_chroma > 0:                                            # the look has a sky palette of its own (used on grey skies)
            st = np.stack([_ab(d.sky_hue, d.sky_chroma * 0.55), _ab(d.sky_hue, d.sky_chroma), _ab(d.sky_hue, d.sky_chroma * 0.9)])
        out = _sky_colour(out, ys, sky.astype(np.float32), st, float(min(1.0, col * 1.0)))
    return out


# ------------------------------------------------------------------------------------------------ light
def _bloom(arr: np.ndarray, d: Drama, k: float) -> np.ndarray:
    """Glow from the brightest areas: blurred highlights added back (screen), tinted by the bloom colour."""
    amt = d.bloom / 100.0 * k
    if amt <= 0:
        return arr
    y = (arr @ LUMA).astype(np.float32)
    h, w = y.shape
    s = min(1.0, 700 / max(h, w))
    ys = cv2.resize(y, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else y
    hi = np.clip((ys - 0.86) / 0.14, 0, 1) ** 1.5
    if float(hi.max()) < 0.05:
        return arr
    e = max(ys.shape)
    g = cv2.GaussianBlur(hi, (0, 0), e * 0.035) * 0.9 + cv2.GaussianBlur(hi, (0, 0), e * 0.12) * 0.6
    g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR) if s < 1 else g
    tint = np.clip(np.array([1.0, 1.0, 1.0], np.float32) * 0.35 + np.append(_hue_rgb(d.bloom_hue), 0)[:3] * 0.65, 0, 1)
    glow = np.clip(g[..., None] * tint * amt * 1.5, 0, 1)
    return 1.0 - (1.0 - arr) * (1.0 - glow)


def _hue_rgb(hue_deg: float) -> np.ndarray:
    lab = np.array([[[75.0, *_ab(hue_deg, 45.0)]]], np.float32)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)[0, 0], 0, 1)


def _mist(arr: np.ndarray, sky: Optional[np.ndarray], depth: Optional[np.ndarray], d: Drama, k: float) -> np.ndarray:
    """A luminous haze lying on the horizon (a band, broken up by soft noise) and atmospheric depth: far ground sinks into haze."""
    amt = d.mist / 100.0 * k
    if amt <= 0 and d.depth_haze <= 0:
        return arr
    h, w = arr.shape[:2]
    col = np.clip(_hue_rgb(d.hi_hue) * 0.30 + 0.78, 0, 1)                   # a pale, slightly coloured haze
    out = arr
    hz = d.mist_y if d.mist_y >= 0 else _horizon(sky)
    if amt > 0 and hz is not None and 0.08 < hz < 0.92 and (d.mist_y >= 0 or (sky is not None and _level_horizon(sky))):
        yy = np.arange(h, dtype=np.float32)[:, None] / h
        band = np.exp(-(((yy - hz) / 0.035) ** 2))
        if depth is not None:                                            # haze lies on DISTANT land only: never across a near tree or rock
            band = band * np.clip((1.0 - depth - 0.15) / 0.5, 0, 1)
        rng = np.random.default_rng(11)
        n = cv2.GaussianBlur(rng.random((max(8, h // 8), max(8, w // 8))).astype(np.float32), (0, 0), 3)
        n = cv2.resize((n - n.min()) / (np.ptp(n) + 1e-6), (w, h), interpolation=cv2.INTER_CUBIC)
        a = (band * (0.6 + 0.4 * n) * amt * 0.6)[..., None]
        out = out * (1 - a) + col * a
    dh = d.depth_haze / 100.0 * k
    if depth is not None and dh > 0:
        far = np.clip((1.0 - depth - 0.30) / 0.70, 0, 1) ** 1.6
        ground = 1.0 - (np.clip(sky, 0, 1) if sky is not None else 0.0)
        a = (far * ground * dh * 0.30)[..., None]
        y = (out @ LUMA)[..., None]
        out = out * (1 - a) + (col * np.clip(0.55 + 0.6 * y, 0.5, 1.0)) * a
    return np.clip(out, 0, 1)


def _anchor_white(arr: np.ndarray, target: float) -> np.ndarray:
    """Give the picture its light back. Highlight recovery (and sky burn) squeeze the top of the tonal range, so the sun and the brightest cloud
    end up grey-ish; a pro keeps real whites. The upper tones are lifted (hue kept) until the brightest ~0.5% reaches the photo's own level."""
    if target <= 0:
        return arr
    y = (arr[::3, ::3] @ LUMA)
    hi = float(np.percentile(y, 99.5))
    if hi >= target - 0.02:
        return arr
    g = target / max(hi, 0.25)
    yy = arr @ LUMA
    w = np.clip((yy - 0.30) / 0.45, 0, 1)
    w = w * w * (3 - 2 * w)                                          # smoothstep: shadows and mids untouched
    return np.clip(arr * (1.0 + (g - 1.0) * w)[..., None], 0, 1)


# ------------------------------------------------------------------------------------------------ the stage
def apply(arr: np.ndarray, masks: Dict[str, np.ndarray], depth: Optional[np.ndarray], d: Drama, amount: float = 1.0) -> np.ndarray:
    """The picture with the Drama look applied. `masks` are at the picture's size ('sky', 'skin' are used); `depth`: 1 = near, or None."""
    k = float(amount) * d.amount / 100.0
    if not d.enabled or k <= 0:
        return arr
    sky = masks.get("sky")
    skin = masks.get("skin")
    ys = _smooth_luma(arr)
    targets = np.stack([_ab(d.sh_hue, d.sh_chroma), _ab(d.mid_hue, d.mid_chroma), _ab(d.hi_hue, d.hi_chroma)])
    prot = np.ones(arr.shape[:2], np.float32)
    if d.skin_safe and skin is not None:
        prot = 1.0 - 0.9 * np.clip(skin, 0, 1)
    if d.sky_chroma > 0 and sky is not None:                                 # the sky takes its own palette, not the land's
        prot = prot * (1.0 - np.clip(sky, 0, 1) * min(1.0, d.sky_colour / 100.0 * k * 1.15))
    out = arr
    mood = d.mood / 100.0 * min(k, 1.3)
    if mood > 0:
        out = _colourise(out, ys, prot, targets, float(min(1.0, mood)), d.protect / 100.0)
    if abs(d.enrich) > 0.5:
        out = _enrich(out, d.enrich / 100.0 * k, prot)
    if sky is not None:
        out = _sky(out, sky, d, k, targets)
    out = _bloom(out, d, k)
    out = _mist(out, sky, depth, d, k)
    out = _anchor_white(out, d.white)
    return np.clip(out, 0, 1).astype(np.float32)
