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
def exposure(arr: np.ndarray, v: float) -> np.ndarray:
    """+-100 = +-3 EV in linear light, with a soft shoulder so highlights roll off instead of clipping."""
    if not v:
        return arr
    lin = np.power(np.clip(arr, 0, 1), GAMMA) * (2.0 ** (v / 100 * 3.0))
    lin = np.where(lin > 0.8, 0.8 + 0.2 * np.tanh((lin - 0.8) / 0.2), lin)
    return np.power(np.clip(lin, 0, 1), 1 / GAMMA)


@banded
def contrast(arr: np.ndarray, v: float) -> np.ndarray:
    if not v:
        return arr
    k = v / 100
    if k > 0:
        s = arr * arr * (3 - 2 * arr)
        return np.clip(arr + (s - arr) * (1.6 * k), 0, 1)
    return np.clip(0.5 + (arr - 0.5) * (1 + 0.75 * k), 0, 1)


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


@banded
def hue_shift(arr: np.ndarray, v: float) -> np.ndarray:
    """+-100 = +-30 degrees around the colour wheel."""
    if not v:
        return arr
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + v / 100 * 30) % 360
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1)


@banded
def mixer(arr: np.ndarray, mix: Dict[str, List[float]]) -> np.ndarray:
    """Lightroom's colour mixer: hue, saturation and luminance per colour family."""
    if not mix or not any(any(v) for v in mix.values()):
        return arr
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    h, s, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    dh, ds, dv = np.zeros_like(h), np.zeros_like(s), np.zeros_like(val)
    for name, (hh, ss, ll) in mix.items():
        if not (hh or ss or ll) or name not in MIX_CENTERS:
            continue
        d = np.abs((h - MIX_CENTERS[name] + 180) % 360 - 180)
        w = np.clip(1 - d / MIX_WIDTH[name], 0, 1) ** 1.3 * np.clip(s * 3, 0, 1)   # ignore near-greys
        dh += w * (hh / 100 * 30)
        ds += w * (ss / 100)
        dv += w * (ll / 100 * 0.6)
    hsv[..., 0] = (h + dh) % 360
    hsv[..., 1] = np.clip(s * (1 + ds), 0, 1)
    hsv[..., 2] = np.clip(val * (1 + dv), 0, 1)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1)


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
