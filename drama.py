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


@banded
def _scale_px(arr: np.ndarray, g: np.ndarray) -> np.ndarray:
    return np.clip(arr * g[..., None], 0, 1)


@banded
def _add_px(arr: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.clip(arr + a[..., None], 0, 1)


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
    out = arr
    burn = d.sky_burn / 100.0 * k
    if burn > 0:
        yy = np.arange(h, dtype=np.float32)[:, None] / (hz * h)
        ramp = np.clip(1.0 - yy, 0, 1) ** 1.15                         # 1 at the top of the frame, 0 at the horizon
        yl = _smooth_luma(out)
        room = np.clip((yl - 0.30) / 0.40, 0, 1)                        # a sky that is already dark or deep is not burned further
        f = 1.0 - 0.34 * burn * ramp * room                             # gentle: a pro lowers a bright sky by ~20%, it must stay luminous
        out = _scale_px(out, 1.0 + (f - 1.0) * sky)
    mdl = d.sky_model / 100.0 * k
    if mdl > 0:
        y = (out @ LUMA).astype(np.float32)
        sel = sky0 > 0.5
        if sel.sum() > 200:
            top = float(np.percentile(y[sel], 97))
            if top > 0.5:
                g = 1.0 + 0.9 * mdl                                       # a gamma around the sky's own white: bright clouds stay white, the veil between them deepens
                yr = np.clip(y / top, 0, 1.2)
                f = np.clip(np.where(y > 1e-4, (top * yr ** g) / np.maximum(y, 1e-4), 1.0), 0.2, 1.0)
                out = _scale_px(out, 1.0 + (f - 1.0) * sky)
    tex = min(d.sky_texture / 100.0 * k, 0.75)
    if tex > 0:
        # computed at a FIXED working size, so the result looks the same at any preview or export size (the cloud shapes are what is boosted)
        sc = min(1.0, 640.0 / max(h, w))
        sw_, sh_ = max(8, int(w * sc)), max(8, int(h * sc))
        ys_ = cv2.resize((out @ LUMA).astype(np.float32), (sw_, sh_), interpolation=cv2.INTER_AREA)
        sm = cv2.resize((sky0 > 0.5).astype(np.float32), (sw_, sh_), interpolation=cv2.INTER_AREA)
        sm = (sm > 0.5).astype(np.float32)
        rs = max(2, int(round(0.012 * max(sw_, sh_))))
        core = cv2.erode(sm.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rs + 1, 2 * rs + 1))).astype(np.float32)
        core = cv2.GaussianBlur(core, (0, 0), rs / 2.0)                          # texture only well inside the sky, never at the skyline
        k5 = np.ones((3, 3), np.uint8)
        busy = ((cv2.dilate(ys_, k5) - cv2.erode(ys_, k5)) > 0.16).astype(np.uint8)  # twigs, wires, leaf edges: strong local contrast, not clouds
        busy = cv2.GaussianBlur(cv2.dilate(busy, np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 1.5)
        core = core * (1.0 - np.clip(busy * 1.4, 0, 1))

        def smooth(sig):                                                        # blur that only looks at SKY pixels (a dark hill must not read as a cloud edge)
            return cv2.GaussianBlur(ys_ * sm, (0, 0), sig) / np.maximum(cv2.GaussianBlur(sm, (0, 0), sig), 1e-3)
        det = (ys_ - smooth(max(2.0, sh_ / 45))) * sm
        fine = (ys_ - smooth(max(1.5, sh_ / 150))) * sm
        soft = lambda x, t: np.sign(x) * np.maximum(np.abs(x) - t, 0.0)         # ignore gradient steps and noise: only real structure
        cloudy = cv2.GaussianBlur(np.abs(soft(det, 0.012)), (0, 0), max(4.0, sh_ / 12))
        cloudy = np.clip((cloudy - 0.006) / 0.02, 0, 1)                         # 0 for a clear sky, 1 for a cloudy one
        boost = (0.17 * np.tanh(soft(det, 0.012) / 0.07) + 0.06 * np.tanh(soft(fine, 0.006) / 0.035)) * core
        boost = cv2.resize((tex * boost * cloudy).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        out = _add_px(out, boost * sky)
    col = d.sky_colour / 100.0 * k
    if col > 0:
        ys = _smooth_luma(out)
        st = targets
        if d.sky_chroma > 0:                                            # the look has a sky palette of its own (used on grey skies)
            st = np.stack([_ab(d.sky_hue, d.sky_chroma * 0.55), _ab(d.sky_hue, d.sky_chroma), _ab(d.sky_hue, d.sky_chroma * 0.9)])
        out = _sky_colour(out, ys, sky.astype(np.float32), st, float(min(1.0, col * 1.0)))
    return out


# ------------------------------------------------------------------------------------------------ light
def _find_sun(arr: np.ndarray):
    """(cx, cy) in pixels of the sun / a lamp, or None. The light is the blown-out (clipped) mass of the ORIGINAL picture (leaf gaps merged): a
    compact one is the light itself; a large one (sun inside a blown sky) has the light at its lowest part, where it meets the horizon or trees."""
    h, w = arr.shape[:2]
    s = min(1.0, 360 / max(h, w))
    small = cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else arr
    y = (small @ LUMA).astype(np.float32)
    blown = (y >= 0.97).astype(np.float32)
    e = max(y.shape)
    merged = (cv2.GaussianBlur(blown, (0, 0), 0.025 * e) > 0.30).astype(np.uint8)       # leaf gaps and twigs do not split the light
    n, lab, st, cen = cv2.connectedComponentsWithStats(merged)
    best, area = None, 0.0
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA] / merged.size
        if 0.0006 < a < 0.12 and a > area and float(blown[lab == i].sum()) > 0.004 * merged.size:
            best, area = i, a
    if best is None:
        return None
    sh, sw = y.shape
    ys, xs = np.nonzero((lab == best) & (blown > 0))
    if len(ys) < 5:
        return None
    if area <= 0.03:
        cx, cy = float(xs.mean()), float(ys.mean())
    else:
        lo = ys > ys.min() + 0.80 * (ys.max() - ys.min())
        cx, cy = float(xs[lo].mean()), float(ys[lo].mean())
    if cy / sh > 0.62:                                                  # a sun / lamp is not in the lowest third (foam, wet rocks and white water are not lights)
        return None
    # a light sits ON something: just below it the picture must fall away (horizon, trees, ground); a bright patch floating in cloud is not a sun
    x0, x1 = int(max(0, cx - 0.08 * sw)), int(min(sw, cx + 0.08 * sw))
    y0, y1 = int(min(sh - 1, cy + 0.02 * sh)), int(min(sh, cy + 0.10 * sh))
    below = float(y[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else 1.0
    if below > 0.72 and area > 0.003:
        return None
    return float(cx / sw * w), float(cy / sh * h)


def _sun(arr: np.ndarray, d: Drama, k: float) -> np.ndarray:
    """A warm glow around the sun and soft rays radiating from it: what makes a backlit scene feel lit rather than merely exposed."""
    ag, ar = d.glow / 100.0 * k, d.rays / 100.0 * k
    if ag <= 0 and ar <= 0:
        return arr
    h, w = arr.shape[:2]
    if d.sun_x < 0 or d.sun_y < 0:
        return arr
    src = (d.sun_x * w, d.sun_y * h)
    s = min(1.0, 720 / max(h, w))
    sh, sw = max(2, int(h * s)), max(2, int(w * s))
    cx, cy = src[0] * s, src[1] * s
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    R = 0.30 * max(sh, sw)
    lab_c = np.array([[[80.0, *_ab(d.bloom_hue, 68.0)]]], np.float32)
    col = np.clip(cv2.cvtColor(lab_c, cv2.COLOR_Lab2RGB)[0, 0], 0, 1)          # a saturated version of the bloom colour for the glow
    out = arr
    add = np.zeros((sh, sw), np.float32)
    if ag > 0:
        add += np.exp(-(r / R) ** 1.25) * ag * 0.85
    if ar > 0:
        rng = np.random.default_rng(7)
        nb = 720
        noise = rng.random(nb).astype(np.float32)
        prof = cv2.GaussianBlur(np.tile(noise, 3)[None, :], (0, 0), 3.0)[0][nb:2 * nb]          # wrap-around: the rays close the circle
        prof = np.clip((prof - np.percentile(prof, 55)) / (np.ptp(prof) * 0.45), 0, 1) ** 1.4
        th = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * nb
        ray = prof[np.clip(th.astype(np.int32), 0, nb - 1)]
        add += ray * np.exp(-r / (1.1 * R)) * (r > 3) * ar * 0.55
    add = cv2.resize(cv2.GaussianBlur(add, (0, 0), 1.2), (w, h), interpolation=cv2.INTER_LINEAR)
    y = (arr @ LUMA)
    add = add * np.clip(0.40 + 1.0 * y, 0, 1)                          # light passes over dark trunks only faintly
    glow = np.clip(add[..., None] * col, 0, 1)
    out = 1.0 - (1.0 - out) * (1.0 - glow)
    if ag > 0:                                                         # the sun itself: a clean, nearly white core
        core = cv2.resize(np.exp(-(r / (0.075 * max(sh, sw))) ** 1.6) * min(1.0, ag * 1.4), (w, h), interpolation=cv2.INTER_LINEAR)
        out = 1.0 - (1.0 - out) * (1.0 - (core[..., None] * np.array([1.0, 0.97, 0.9], np.float32)))
    return out


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
    med = float(np.median(arr[::6, ::6] @ LUMA))
    lit = float(np.clip((med - 0.10) / 0.22, 0, 1))                         # a night scene has no luminous haze: a pale band across darkness looks like a smear
    amt *= lit
    d = d.model_copy(update={'depth_haze': d.depth_haze * lit})
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
        dark = np.clip(((out @ LUMA) - 0.07) / 0.20, 0, 1)                   # no luminous haze over very dark near objects (a tree in front of the horizon)
        a = (band * (0.6 + 0.4 * n) * amt * 0.6 * dark)[..., None]
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
    return _scale_px(arr, 1.0 + (g - 1.0) * w)


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
    out = _sun(out, d, k)
    out = _mist(out, sky, depth, d, k)
    out = _anchor_white(out, d.white)
    return np.clip(out, 0, 1).astype(np.float32)
