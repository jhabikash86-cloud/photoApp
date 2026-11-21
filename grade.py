"""Lightroom-style grading operations with strong mappings.

Each operation takes a float32 RGB array in [0,1] (gamma encoded) and a slider value in -100..+100, and
returns a new array. Mappings are deliberately powerful, so a small slider move is clearly visible, as in
Lightroom. Tone weights come from an edge-aware blurred luminance, so adjustments do not halo.
"""
from typing import Dict, List, Optional

import cv2
import numpy as np

import colorcal
import landscape
from adjust import edge_aware_blur
from bands import banded

GAMMA = 2.2
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
MIX_CENTERS = {"red": 0, "orange": 30, "yellow": 60, "green": 120, "aqua": 180, "blue": 240, "purple": 275, "magenta": 320}
MIX_WIDTH = {"red": 35, "orange": 28, "yellow": 35, "green": 55, "aqua": 45, "blue": 50, "purple": 40, "magenta": 40}


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


class Ctx:
    """Lazily computed per-image fields shared by many operations."""

    def __init__(self, arr: np.ndarray, exclude: Optional[np.ndarray] = None):
        self.arr = arr
        self.edge = max(arr.shape[:2])
        self.exclude = exclude      # e.g. the sky: a clear sky is not haze, dehazing it only makes smudges
        self._f: Dict[str, object] = {}

    def _get(self, key, fn):
        if key not in self._f:
            self._f[key] = fn()
        return self._f[key]

    @property
    def y(self) -> np.ndarray:
        return self._get("y", lambda: (self.arr @ LUMA).astype(np.float32))

    @property
    def yb(self) -> np.ndarray:        # edge-aware, wide: tone-range membership
        return self._get("yb", lambda: np.clip(edge_aware_blur(self.y, self.edge / 40, 0.03), 0, 1))

    @property
    def detail_c(self) -> np.ndarray:  # clarity band
        return self._get("dc", lambda: self.y - edge_aware_blur(self.y, self.edge / 150, 0.05))

    @property
    def detail_t(self) -> np.ndarray:  # texture band
        return self._get("dt", lambda: self.y - edge_aware_blur(self.y, max(2.0, self.edge / 500), 0.05))

    @property
    def haze(self) -> tuple:
        def build():
            t, A = landscape.dehaze_fields(self.arr)
            if self.exclude is not None:
                t = 1 - (1 - t) * (1 - 0.9 * np.clip(self.exclude, 0, 1))
            return t, A
        return self._get("haze", build)

    def band(self, y0: int, y1: int) -> "Ctx":
        """Rows y0..y1 of this context, reading the shared fields lazily (nothing is recomputed per band)."""
        return _BandCtx(self, y0, y1)

    def crop(self, box) -> "Ctx":
        """A view of this context limited to box=(y0,y1,x0,x1); heavy fields are computed once, then sliced."""
        y0, y1, x0, x1 = box
        c = Ctx(self.arr[y0:y1, x0:x1])
        c.edge = self.edge
        c.exclude = None if self.exclude is None else self.exclude[y0:y1, x0:x1]
        for name in ("y", "yb", "detail_c", "detail_t"):
            c._f[{"y": "y", "yb": "yb", "detail_c": "dc", "detail_t": "dt"}[name]] = getattr(self, name)[y0:y1, x0:x1]
        t, A = self.haze
        c._f["haze"] = (t[y0:y1, x0:x1], A)
        return c


class _BandCtx(Ctx):
    def __init__(self, parent: Ctx, y0: int, y1: int):
        super().__init__(parent.arr[y0:y1])
        self.edge = parent.edge
        self._p, self._s = parent, slice(y0, y1)
        self.exclude = None if parent.exclude is None else parent.exclude[self._s]

    y = property(lambda self: self._p.y[self._s])
    yb = property(lambda self: self._p.yb[self._s])
    detail_c = property(lambda self: self._p.detail_c[self._s])
    detail_t = property(lambda self: self._p.detail_t[self._s])

    @property
    def haze(self) -> tuple:
        t, A = self._p.haze
        return t[self._s], A


# ----------------------------------------------------------------------------------------------
# single operations
# ----------------------------------------------------------------------------------------------
@banded
@banded
def exposure(arr: np.ndarray, v: float) -> np.ndarray:
    """+-100 = +-3 EV in linear light. The brightest channel gets the soft shoulder and the other channels follow it (the colour ratios are kept), so a vivid
    blue or red brightens as the same blue or red instead of clipping one channel at a time and skewing in hue; the over-range part drifts a little toward
    white, the way film and raw converters roll highlights off."""
    if not v:
        return arr
    lin = np.power(np.clip(arr, 0, 1), GAMMA) * (2.0 ** (v / 100 * 3.0))
    m = lin.max(axis=2, keepdims=True)
    m2 = np.where(m > 0.8, 0.8 + 0.2 * np.tanh((m - 0.8) / 0.2), m)
    out = lin * (m2 / np.maximum(m, 1e-6))
    over = np.clip((m - 0.8) / 1.6, 0, 1) * 0.30                                   # blown colours drift toward white, never to a hue-shifted clip
    out = out * (1 - over) + m2 * over
    return np.power(np.clip(out, 0, 1), 1 / GAMMA)


def _limit_to_gamut(out: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pull any colour that went outside 0..1 back toward its own luminance (hue and brightness kept) instead of clipping a channel."""
    mx, mn = out.max(axis=2, keepdims=True), out.min(axis=2, keepdims=True)
    t = np.ones_like(mx)
    hi = mx > 1.0
    t = np.where(hi, np.minimum(t, (1.0 - y) / np.maximum(mx - y, 1e-6)), t)
    lo = mn < 0.0
    t = np.where(lo, np.minimum(t, y / np.maximum(y - mn, 1e-6)), t)
    return np.clip(y + (out - y) * np.clip(t, 0, 1), 0, 1)


@banded
def contrast(arr: np.ndarray, v: float) -> np.ndarray:
    """An S-curve on the LUMINANCE; the colour follows. Curving each channel on its own changes saturation and hue (positive contrast turns skies teal and
    reds orange); here the curve decides the brightness and every pixel keeps its own colour direction."""
    if not v:
        return arr
    k = v / 100
    y = (arr @ LUMA)[..., None]
    if k > 0:
        s = y * y * (3 - 2 * y)
        yt = np.clip(y + (s - y) * (1.6 * k), 0, 1)
    else:
        yt = np.clip(0.5 + (y - 0.5) * (1 + 0.75 * k), 0, 1)
    ratio = yt / np.maximum(y, 1e-3)
    out = arr * np.where(y > 1e-3, ratio, 1.0)
    out = np.where(y > 1e-3, out, arr + (yt - y))
    # chroma follows lightness the way it does in a print: a darker colour keeps its saturation, a lighter one keeps it too (colour difference scaled by the same ratio)
    return _limit_to_gamut(out, yt)


@banded
def highlights(arr: np.ndarray, v: float, ctx: Ctx) -> np.ndarray:
    if not v:
        return arr
    w = smoothstep(0.35, 1.0, ctx.yb)
    scale = 1 + (v / 100) * (0.65 if v < 0 else 0.40) * w
    out = arr * scale[..., None]
    if v < 0:   # recovering highlights must keep the colour: darker blue sky, not grey sky
        gray = (out @ LUMA)[..., None]
        out = gray + (out - gray) * (1 + (-v / 100) * 0.55 * w[..., None])
    return np.clip(out, 0, 1)


@banded
def shadows(arr: np.ndarray, v: float, ctx: Ctx) -> np.ndarray:
    if not v:
        return arr
    w = 1 - smoothstep(0.0, 0.65, ctx.yb)
    scale = 1 + (v / 100) * (1.3 if v > 0 else 0.7) * w
    return np.clip(arr * scale[..., None], 0, 1)


@banded
def whites(arr: np.ndarray, v: float, ctx: Ctx) -> np.ndarray:
    if not v:
        return arr
    return np.clip(arr + ((v / 100) * 0.22 * smoothstep(0.6, 1.0, ctx.y))[..., None], 0, 1)


@banded
def blacks(arr: np.ndarray, v: float, ctx: Ctx) -> np.ndarray:
    if not v:
        return arr
    return np.clip(arr + ((v / 100) * 0.18 * (1 - smoothstep(0.0, 0.35, ctx.y)))[..., None], 0, 1)


def dehaze(arr: np.ndarray, v: float, ctx: Ctx) -> np.ndarray:
    if not v:
        return arr
    return landscape.dehaze_apply(arr, ctx.haze, v / 100)


def _soft_limit(d: np.ndarray, limit: float) -> np.ndarray:
    """Cap the size of a local-contrast correction. Without a cap, a dark ridge against bright sky pushes the sky next to
    it brighter and the ridge darker by a lot: the halo that makes local contrast look like cheap HDR."""
    return limit * np.tanh(d / limit)


@banded
def clarity(arr: np.ndarray, v: float, ctx: Ctx) -> np.ndarray:
    if not v:
        return arr
    y = ctx.y
    d = _soft_limit(ctx.detail_c, 0.045)
    return np.clip(arr + ((v / 100) * 1.8 * d * 4 * y * (1 - y))[..., None], 0, 1)


@banded
def texture(arr: np.ndarray, v: float, ctx: Ctx) -> np.ndarray:
    if not v:
        return arr
    return np.clip(arr + ((v / 100) * 1.5 * _soft_limit(ctx.detail_t, 0.035))[..., None], 0, 1)


def _gamut_limit(gray: np.ndarray, d: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Soft-limit a colour boost f so that gray + f*d stays inside 0..1 (no channel clips): saturated skies and
    flowers keep their detail instead of flattening to a solid colour."""
    up = np.where(d > 1e-6, (1 - gray) / np.maximum(d, 1e-6), 1e6)
    dn = np.where(d < -1e-6, gray / np.maximum(-d, 1e-6), 1e6)
    fmax = np.maximum(np.minimum(up, dn).min(axis=2, keepdims=True), 1.0)
    t = f / fmax
    soft = fmax * np.where(t <= 0.85, t, 0.85 + 0.15 * np.tanh((t - 0.85) / 0.15))
    return np.where(f > 1, soft, f)


@banded
def saturation(arr: np.ndarray, v: float) -> np.ndarray:
    if not v:
        return arr
    gray = (arr @ LUMA)[..., None]
    d = arr - gray
    f = np.full(gray.shape, max(0.0, 1 + v / 100), np.float32)
    if v > 0:
        f = _gamut_limit(gray, d, f)
    return np.clip(gray + d * f, 0, 1)


def _skin_like(arr: np.ndarray) -> np.ndarray:
    """Skin-like hues (orange, moderate saturation) get most of a vibrance boost held back. The weight is smooth, so it is
    computed at a quarter of the size and scaled up (several times cheaper, visually identical)."""
    h0, w0 = arr.shape[:2]
    k = 4 if max(h0, w0) > 700 else 1
    small = cv2.resize(arr, (max(1, w0 // k), max(1, h0 // k)), interpolation=cv2.INTER_AREA) if k > 1 else arr
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hue_w = np.clip(1 - np.abs(hsv[..., 0] - 25) / 22, 0, 1)
    sat_w = np.clip((hsv[..., 1] - 0.12) / 0.15, 0, 1) * np.clip((0.7 - hsv[..., 1]) / 0.2, 0, 1)
    skin_like = (hue_w * sat_w).astype(np.float32)
    if k > 1:
        skin_like = cv2.resize(skin_like, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return skin_like


def vibrance(arr: np.ndarray, v: float, protect: Optional[np.ndarray] = None) -> np.ndarray:
    """Lightroom-style vibrance: lifts muted colours much more than already-saturated ones, and holds back
    on skin-tone hues (and on `protect`, e.g. the detected skin) so people don't turn orange."""
    if not v:
        return arr
    skin_like = _skin_like(arr) if v > 0 else None        # whole-picture field, computed once, then handed to the bands
    return _vibrance(arr, v, protect, skin_like)


@banded
def _vibrance(arr: np.ndarray, v: float, protect: Optional[np.ndarray], skin_like: Optional[np.ndarray]) -> np.ndarray:
    gray = (arr @ LUMA)[..., None]
    chroma = (arr.max(axis=2) - arr.min(axis=2))[..., None]
    if v > 0:
        f = 1 + (v / 100) * 1.9 * (1 - chroma) ** 1.25
        f = 1 + (f - 1) * (1 - 0.75 * skin_like[..., None])
    else:
        f = 1 + (v / 100) * (0.4 + 0.8 * chroma)
    if protect is not None:
        f = 1 + (f - 1) * (1 - 0.92 * protect[..., None])
    f = np.maximum(f, 0)
    if v > 0:
        f = _gamut_limit(gray, arr - gray, f)
    return np.clip(gray + (arr - gray) * f, 0, 1)


def temp_tint(arr: np.ndarray, t: float, ti: float = 0.0) -> np.ndarray:
    if not t and not ti:
        return arr
    return colorcal.apply_gains(arr, np.ones(3, np.float32), t, ti)


# ---------------------------------------------------------------------------------------------
# Perceptual colour tools (OKLCH): the mixer and the hue shift
# ---------------------------------------------------------------------------------------------
# Where each colour family sits on the OKLCH hue circle (degrees). Hue windows overlap smoothly and add up to 1 everywhere, so a family follows its slider
# evenly and never leaks into colours that are not its neighbours.
OK_CENTERS = [("red", 29.0), ("orange", 58.0), ("yellow", 104.0), ("green", 142.0), ("aqua", 196.0), ("blue", 264.0), ("purple", 304.0), ("magenta", 335.0)]
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566], [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099], [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
_M1I, _M2I = np.linalg.inv(_M1).astype(np.float32), np.linalg.inv(_M2).astype(np.float32)


def _to_oklab(arr: np.ndarray) -> np.ndarray:
    lin = np.power(np.clip(arr, 0, 1), GAMMA)
    return np.cbrt(lin @ _M1.T) @ _M2.T


def _from_oklab(lab: np.ndarray) -> np.ndarray:
    lms = lab @ _M2I.T
    return np.power(np.clip((lms * lms * lms) @ _M1I.T, 0, 1), 1 / GAMMA)


def _family_weights(h_deg: np.ndarray) -> dict:
    """Smooth partition of unity over the hue circle: {family: weight 0..1}."""
    names = [n for n, _ in OK_CENTERS]
    cs = np.array([c for _, c in OK_CENTERS], np.float32)
    out = {n: np.zeros_like(h_deg) for n in names}
    k = len(cs)
    for i in range(k):
        a, b = cs[i], cs[(i + 1) % k]
        span = (b - a) % 360
        pos = ((h_deg - a) % 360) / span
        inside = pos <= 1.0
        t = np.clip(pos, 0, 1)
        t = t * t * (3 - 2 * t)                           # smoothstep: no kinks at the family centres
        out[names[i]] = out[names[i]] + np.where(inside, 1 - t, 0)
        out[names[(i + 1) % k]] = out[names[(i + 1) % k]] + np.where(inside, t, 0)
    return out


def _lch_adjust(arr: np.ndarray, hue_deg, sat_mul, lum_add, chroma_gate: bool = True) -> np.ndarray:
    """Apply per-pixel (or scalar) hue rotation (degrees), chroma multiplier and lightness addition in OKLCH. Near-greys are left alone."""
    lab = _to_oklab(arr)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    H = np.arctan2(b, a)
    gate = np.clip((C - 0.012) / 0.05, 0, 1) if chroma_gate else 1.0
    H2 = H + np.radians(hue_deg) * gate
    C2 = C * (1 + (np.asarray(sat_mul) * gate))
    L2 = L + np.asarray(lum_add) * gate * np.clip(0.35 + 2.0 * C, 0, 1) * (1 - 0.6 * np.clip(np.abs(L - 0.5) * 2, 0, 1) ** 3)
    lab2 = np.stack([np.clip(L2, 0, 1), C2 * np.cos(H2), C2 * np.sin(H2)], axis=-1).astype(np.float32)
    out = _from_oklab(lab2)
    y = (arr @ LUMA)[..., None]
    yo = (out @ LUMA)[..., None]
    return np.clip(out, 0, 1)


@banded
def hue_shift(arr: np.ndarray, v: float) -> np.ndarray:
    """+-100 = +-30 degrees around the (perceptual) colour wheel, brightness kept."""
    if not v:
        return arr
    return _lch_adjust(arr, v / 100 * 30.0, 0.0, 0.0)


@banded
def mixer(arr: np.ndarray, mix: Dict[str, List[float]]) -> np.ndarray:
    """Lightroom's colour mixer in a perceptual space. Each family (red ... magenta) has hue (+-100 = +-30 degrees), saturation (-100 = grey, +100 = twice the
    colour) and luminance (+-100 = +-0.42 of lightness). The families overlap smoothly and sum to one, so every colour responds evenly."""
    if not mix or not any(any(v) for v in mix.values()):
        return arr
    lab = _to_oklab(arr)
    H = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
    w = _family_weights(H)
    dh = np.zeros_like(H); ds = np.zeros_like(H); dl = np.zeros_like(H)
    for name, (hh, ss, ll) in mix.items():
        if name not in w or not (hh or ss or ll):
            continue
        dh += w[name] * (hh / 100 * 30.0)
        ds += w[name] * (ss / 100)
        dl += w[name] * (ll / 100 * 0.42)
    return _lch_adjust(arr, dh, ds, dl)


def _hue_rgb(h: float) -> np.ndarray:
    c = cv2.cvtColor(np.array([[[h % 360, 1.0, 1.0]]], np.float32), cv2.COLOR_HSV2RGB)[0, 0]
    return (c - float(c @ LUMA)).astype(np.float32)          # a pure tint: no brightness change


@banded
def apply_wheels(arr: np.ndarray, g, protect: Optional[np.ndarray] = None, protect_sky: Optional[np.ndarray] = None) -> np.ndarray:
    """Colour grading wheels. Each range adds a tint (hue, saturation) and a small luminance shift, weighted by
    where the pixel sits in the tonal range. `protect` (0..1, e.g. skin) shields the mids and highlights tints."""
    ranges = (("shadows", g.shadows), ("mids", g.mids), ("highs", g.highs), ("glob", g.glob))
    if not any(w.sat or w.lum for _, w in ranges):
        return arr
    y = (arr @ LUMA)[..., None]
    soft = 0.35 + 0.9 * g.blending / 100                    # wider overlap with higher blending
    bal = g.balance / 100 * 0.25
    ys = np.clip(y - bal, 0, 1)
    wts = {
        "shadows": np.clip((1 - ys) ** (1.6 / soft + 0.4), 0, 1),
        "highs": np.clip(ys ** (1.6 / soft + 0.4), 0, 1),
        "mids": np.clip(1 - np.abs(ys * 2 - 1) ** (1.2 / soft + 0.3), 0, 1),
        "glob": np.ones_like(y),
    }
    out = arr.copy()
    for name, w in ranges:
        if not (w.sat or w.lum):
            continue
        weight = wts[name]
        if protect is not None and name in ("mids", "highs"):
            weight = weight * (1 - 0.8 * protect[..., None])
        if protect_sky is not None and name in ("mids", "highs"):
            weight = weight * (1 - 0.7 * protect_sky[..., None])   # a warm tint on pale blue sky only makes grey-green mud
        if w.sat:
            out += weight * (w.sat / 100) * 0.30 * _hue_rgb(w.hue)
        if w.lum:
            out += weight * (w.lum / 100) * 0.12
    return np.clip(out, 0, 1)


def _pchip(xs: np.ndarray, ys: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Monotone cubic interpolation (Fritsch-Carlson): a smooth curve through the points that never overshoots between them."""
    n = len(xs)
    if n == 1:
        return np.full_like(xq, ys[0])
    h = np.diff(xs)
    d = np.diff(ys) / np.maximum(h, 1e-9)
    m = np.zeros(n)
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, n - 1):
        m[i] = 0.0 if d[i - 1] * d[i] <= 0 else 3 * (h[i - 1] + h[i]) / ((2 * h[i] + h[i - 1]) / d[i - 1] + (h[i] + 2 * h[i - 1]) / d[i])
    idx = np.clip(np.searchsorted(xs, xq) - 1, 0, n - 2)
    t = (xq - xs[idx]) / np.maximum(h[idx], 1e-9)
    t2, t3 = t * t, t * t * t
    out = (2 * t3 - 3 * t2 + 1) * ys[idx] + (t3 - 2 * t2 + t) * h[idx] * m[idx] + (-2 * t3 + 3 * t2) * ys[idx + 1] + (t3 - t2) * h[idx] * m[idx + 1]
    return np.clip(out, 0.0, 1.0)


def curve_lut(points, size: int = 1024) -> Optional[np.ndarray]:
    """A lookup table for a list of (x, y) curve points, or None when the curve is the identity."""
    pts = sorted((min(1.0, max(0.0, float(p[0]))), min(1.0, max(0.0, float(p[1])))) for p in points)
    uniq = []
    for x, y in pts:
        if uniq and abs(x - uniq[-1][0]) < 1e-4:
            continue
        uniq.append((x, y))
    if len(uniq) < 2 or all(abs(x - y) < 1e-4 for x, y in uniq):
        return None
    if uniq[0][0] > 0:
        uniq.insert(0, (0.0, uniq[0][1]))
    if uniq[-1][0] < 1:
        uniq.append((1.0, uniq[-1][1]))
    xs, ys = np.array([u[0] for u in uniq]), np.array([u[1] for u in uniq])
    return _pchip(xs, ys, np.linspace(0, 1, size)).astype(np.float32)


@banded
def point_curve(arr: np.ndarray, tone) -> np.ndarray:
    """Point tone curves: the master RGB curve first, then the red, green and blue curves."""
    luts = [curve_lut(tone.rgb), curve_lut(tone.r), curve_lut(tone.g), curve_lut(tone.b)]
    if all(l is None for l in luts):
        return arr
    grid = np.linspace(0, 1, 1024, dtype=np.float32)
    out = arr
    if luts[0] is not None:
        out = np.interp(out, grid, luts[0]).astype(np.float32)
    for c, l in enumerate(luts[1:]):
        if l is not None:
            out = out.copy() if out is arr else out
            out[..., c] = np.interp(out[..., c], grid, l)
    return np.clip(out, 0, 1)


@banded
def parametric_curve(arr: np.ndarray, c) -> np.ndarray:
    """Parametric tone curve (Lightroom style). Each slider (-100..100) bends the curve inside one tonal range:
    shadows, darks, lights, highlights. The curve stays monotonic, so it can never invert tones or posterise."""
    vals = (c.shadows, c.darks, c.lights, c.highlights)
    if not any(vals):
        return arr
    x = np.linspace(0, 1, 513, dtype=np.float32)
    centers = (0.12, 0.37, 0.63, 0.88)
    dy = np.zeros_like(x)
    for v, cen in zip(vals, centers):
        dy += (v / 100) * 0.14 * np.exp(-((x - cen) / 0.16) ** 2)
    y = np.clip(x + dy, 0, 1)
    y = np.maximum.accumulate(y)                      # monotonic: tones never swap order
    y = (y - y[0]) / max(y[-1] - y[0], 1e-6)          # keep black at black and white at white
    return np.interp(arr, x, y).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# groups
# ----------------------------------------------------------------------------------------------
def apply_basic(arr: np.ndarray, b, mix=None, skin_protect: Optional[np.ndarray] = None,
                haze_exclude: Optional[np.ndarray] = None) -> tuple:
    """Whole-picture adjustments in Lightroom's order. Returns (result, context of the result)."""
    out = exposure(arr, b.exposure)
    out = contrast(out, b.contrast)
    ctx = Ctx(out, haze_exclude)       # tone ranges follow the picture after exposure/contrast
    out = highlights(out, b.highlights, ctx)
    out = shadows(out, b.shadows, ctx)
    if b.whites or b.blacks:
        ctx = Ctx(out, haze_exclude)
        out = whites(out, b.whites, ctx)
        out = blacks(out, b.blacks, ctx)
    if b.dehaze or b.clarity or b.texture:
        ctx = Ctx(out, haze_exclude)
        out = dehaze(out, b.dehaze, ctx)
        if b.clarity or b.texture:
            ctx = Ctx(out, haze_exclude)
            out = clarity(out, b.clarity, ctx)
            out = texture(out, b.texture, ctx)
    out = mixer(out, mix)
    out = vibrance(out, b.vibrance, skin_protect)
    out = saturation(out, b.saturation)
    return out, Ctx(out, haze_exclude)


def region_delta(arr: np.ndarray, ctx: Ctx, r, mask: np.ndarray):
    """What a local adjustment adds to the picture inside `mask`: ((y0, y1, x0, x1), patch), or None if there
    is nothing to do. Works only on the mask's bounding box, so small regions are cheap in time and memory."""
    hl, sh_, tint = getattr(r, 'highlights', 0), getattr(r, 'shadows', 0), getattr(r, 'tint', 0)
    if not any((r.exposure, r.contrast, r.vibrance, r.saturation, r.clarity, r.dehaze, r.temperature, r.hue, hl, sh_, tint)):
        return None
    ys, xs = np.nonzero(mask > 0.02)
    if len(ys) < 64:
        return None
    pad = 8
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    sub = arr[y0:y1, x0:x1]
    c = ctx.crop((y0, y1, x0, x1))
    m = mask[y0:y1, x0:x1]
    f = exposure(sub, r.exposure)
    f = contrast(f, r.contrast)
    if hl or sh_:
        f = highlights(f, hl, c)
        f = shadows(f, sh_, c)
    f = temp_tint(f, r.temperature, tint)
    f = hue_shift(f, r.hue)
    f = vibrance(f, r.vibrance)
    f = saturation(f, r.saturation)
    if r.clarity:
        f = clarity(f, r.clarity, c)
    if r.dehaze:
        f = landscape.dehaze_apply(f, c.haze, r.dehaze / 100, m)
    return (y0, y1, x0, x1), ((f - sub) * m[..., None]).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# effects
# ----------------------------------------------------------------------------------------------
def vignette(arr: np.ndarray, amount: float, mid: float = 50) -> np.ndarray:
    if not amount:
        return arr
    H, W = arr.shape[:2]
    sub = max(1, max(H, W) // 600)
    ys, xs = np.mgrid[0:H:sub, 0:W:sub].astype(np.float32)
    r = np.hypot((xs - W / 2) / (W / 2), (ys - H / 2) / (H / 2)) / 1.41
    d = smoothstep(0.25 + 0.5 * mid / 100, 1.05, r)
    d = cv2.resize(d.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    k = amount / 100
    if k < 0:
        return np.clip(arr * (1 + k * 0.85 * d), 0, 1)
    return np.clip(1 - (1 - arr) * (1 - k * 0.7 * d), 0, 1)


def grain(arr: np.ndarray, amount: float) -> np.ndarray:
    if not amount:
        return arr
    H, W = arr.shape[:2]
    rng = np.random.default_rng(7)
    n = rng.normal(0, 1, (H, W)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), max(0.6, max(H, W) / 2500))
    y = arr @ LUMA
    weight = (4 * y * (1 - y) * 0.8 + 0.2)[..., None]        # film grain lives in the midtones
    return np.clip(arr + n[..., None] * weight * (amount / 100) * 0.07 * 3, 0, 1)
