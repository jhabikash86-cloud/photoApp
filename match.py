"""Match a reference: measure the look of a reference picture and move the sliders so your photo gets the same look.

The result is ordinary, editable slider values (a tone curve, grading wheels, saturation / vibrance, colour-mixer entries),
not a hidden filter. The solve is closed-loop: render the current edit, measure how far it still is from the reference, nudge
the sliders, repeat. A strength value (0..100) blends between your original sliders and the fully matched ones.

What is matched: brightness distribution (blacks, shadows, mids, highlights), the colour cast of shadows / mids / highlights,
overall colour intensity, and how strong / which hue each colour family (reds, greens, blues ...) is.
What is not copied: content, composition, or local retouching: only the global look.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

import grade
from params import EditParams, MIXER_COLORS

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
QS = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
CURVE_X = [0.0, 0.06, 0.15, 0.3, 0.5, 0.7, 0.85, 0.94, 1.0]
HUE_CENTERS = {"red": 0, "orange": 30, "yellow": 60, "green": 120, "aqua": 180, "blue": 225, "purple": 275, "magenta": 320}
RANGES = ("shadows", "mids", "highs")
WHEEL_GAIN = 1.0 / 22.0          # one wheel-saturation unit (0..1) per ~22 Lab chroma; refined by the closed loop


@dataclass
class Look:
    q: np.ndarray                                   # luma quantiles
    chroma: float                                   # mean chroma of coloured pixels
    ab: dict                                        # tonal range -> mean (a, b) cast
    fam: dict                                       # hue family -> (fraction, mean chroma, mean hue offset in degrees)


@dataclass
class Match:
    ref_id: str
    base_key: str
    full: EditParams
    base: EditParams
    notes: List[str] = field(default_factory=list)
    residual: float = 0.0


def measure(rgb: np.ndarray) -> Look:
    """Look statistics of an RGB float picture (0..1)."""
    h, w = rgb.shape[:2]
    s = 420 / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    y = rgb @ LUMA
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    a, b = lab[..., 1], lab[..., 2]
    c = np.hypot(a, b)
    q = np.quantile(y, QS)
    coloured = c > 6
    chroma = float(c[coloured].mean()) if coloured.mean() > 0.02 else float(c.mean())
    ab = {}
    for name, wt in (("shadows", np.clip(1 - y / 0.4, 0, 1)), ("mids", np.clip(1 - np.abs(y - 0.5) / 0.3, 0, 1)),
                     ("highs", np.clip((y - 0.55) / 0.35, 0, 1))):
        wt = wt * np.clip(c / 12.0, 0.15, 1.0)         # near-greys carry little information about a cast
        tot = float(wt.sum()) + 1e-6
        ab[name] = (float((a * wt).sum() / tot), float((b * wt).sum() / tot))
    hue = (np.degrees(np.arctan2(b, a)) % 360)
    hsv_h = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 0]
    fam = {}
    for name, centre in HUE_CENTERS.items():
        d = np.abs((hsv_h - centre + 180) % 360 - 180)
        m = (d < 22) & (c > 8)
        frac = float(m.mean())
        if frac > 0.004:
            off = float(np.mean((hsv_h[m] - centre + 180) % 360 - 180))
            fam[name] = (frac, float(c[m].mean()), off)
    return Look(q=q, chroma=chroma, ab=ab, fam=fam)


# ------------------------------------------------------------------------------------------------ helpers
def _dir(hue_deg: float) -> np.ndarray:
    """Unit direction in Lab (a, b) of a wheel hue."""
    rgb = cv2.cvtColor(np.array([[[hue_deg % 360, 1.0, 1.0]]], np.float32), cv2.COLOR_HSV2RGB)
    lab = cv2.cvtColor(np.clip(rgb - (rgb @ LUMA)[..., None] + 0.5, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)[0, 0]
    v = np.array([lab[1], lab[2]], np.float32)
    return v / (np.linalg.norm(v) + 1e-6)


def _wheel_vec(w) -> np.ndarray:
    return _dir(w.hue) * (w.sat / 100.0)


WHEEL_CAP = {"shadows": 40.0, "mids": 28.0, "highs": 35.0}      # wheels beyond this go garish; a reference's strong cast is only approached


def _set_wheel(w, vec: np.ndarray, cap: float = 40.0):
    n = float(np.linalg.norm(vec))
    if n < 0.01:
        w.sat = 0
        return
    # find the hue whose direction matches vec (coarse search over 5 degrees, then local refine)
    best, bd = 0.0, -2.0
    u = vec / n
    for hh in range(0, 360, 5):
        d = float(_dir(hh) @ u)
        if d > bd:
            best, bd = float(hh), d
    for hh in np.arange(best - 5, best + 5.01, 1.0):
        d = float(_dir(hh) @ u)
        if d > bd:
            best, bd = float(hh), d
    w.hue = best % 360
    w.sat = float(min(cap, n * 100.0))


def _tone_correction(cur_q: np.ndarray, ref_q: np.ndarray):
    """A function x -> x' that moves the current luma distribution onto the reference one (monotone, gently limited)."""
    xs = np.concatenate([[0.0], cur_q, [1.0]])
    ys = np.concatenate([[0.0 if ref_q[0] > 0.02 else 0.0], ref_q, [1.0]])
    # keep strictly increasing
    for i in range(1, len(xs)):
        xs[i] = max(xs[i], xs[i - 1] + 1e-3)
        ys[i] = max(ys[i], ys[i - 1] + 1e-3)
    ys = np.minimum(ys, 1.0)
    # the black point of the reference: lifted blacks (matte) are a real style, so pin the end where the reference sits
    ys[0] = float(np.clip(ref_q[0] - cur_q[0] * 0.9, 0.0, 0.18))   # the 1% point sits above pure black: extend the same shift down to 0
    xs[0] = 0.0
    ys[-1] = 1.0
    return xs, ys


def _slope_limit(xs, ys, lo=0.35, hi=2.6):
    ys = np.array(ys, np.float64)
    for _ in range(3):
        for i in range(1, len(xs)):
            dx = xs[i] - xs[i - 1]
            ys[i] = ys[i - 1] + float(np.clip(ys[i] - ys[i - 1], lo * dx, hi * dx))
    return np.clip(ys, 0, 1)


def _hue_name(deg: float) -> str:
    names = [(15, "red"), (45, "orange"), (75, "yellow"), (150, "green"), (200, "teal"), (255, "blue"), (300, "purple"), (345, "pink"), (361, "red")]
    for lim, n in names:
        if deg < lim:
            return n
    return "red"


def _cast_name(ab) -> str:
    n = math.hypot(*ab)
    if n < 2.5:
        return "neutral"
    deg = (math.degrees(math.atan2(ab[1], ab[0])) % 360)
    # Lab a/b angle: 0 = magenta-red, 90 = yellow, 180 = green, 270 = blue
    labels = [(30, "magenta-red"), (80, "warm orange"), (110, "yellow"), (160, "yellow-green"), (210, "green"), (250, "teal"), (300, "blue"), (340, "purple"), (361, "magenta-red")]
    for lim, name in labels:
        if deg < lim:
            return name
    return "magenta-red"


# ------------------------------------------------------------------------------------------------ the solve
def solve(render_small, base: EditParams, ref: Look, iterations: int = 3) -> Match:
    """`render_small(params) -> RGB float picture` renders the photo with the given sliders (small, fast)."""
    p = base.model_copy(deep=True)
    notes: List[str] = []
    resid = 0.0
    first = measure(render_small(p))

    for it in range(iterations):
        cur = measure(render_small(p))

        # ---- 1. brightness distribution -> master curve (composed with whatever curve is already there)
        xs, ys = _tone_correction(cur.q, ref.q)
        corr_y = _slope_limit(xs, ys)
        gain = 1.0 if it == 0 else 0.7
        existing = grade.curve_lut(p.tone.rgb)
        grid = np.linspace(0, 1, 1024, dtype=np.float32)
        pts = []
        for x in CURVE_X:
            ex = float(np.interp(x, grid, existing)) if existing is not None else x
            corrected = float(np.interp(ex, xs, corr_y))
            pts.append([x, float(np.clip(ex + (corrected - ex) * gain, 0, 1))])
        pts[0][0], pts[-1][0] = 0.0, 1.0
        pts[-1][1] = 1.0
        # make it a valid, monotone curve
        for i in range(1, len(pts)):
            pts[i][1] = max(pts[i][1], pts[i - 1][1] + 0.002)
        pts[-1][1] = min(1.0, pts[-1][1])
        p.tone.rgb = [[round(a, 4), round(b, 4)] for a, b in pts]

        # ---- 2. colour cast in shadows / mids / highlights -> grading wheels
        cur = measure(render_small(p))
        for name in RANGES:
            d = np.array(ref.ab[name]) - np.array(cur.ab[name])
            w = getattr(p.grading, name)
            v = _wheel_vec(w) + d * WHEEL_GAIN * (0.9 if name != "mids" else 0.6)
            _set_wheel(w, v, WHEEL_CAP[name])

        # ---- 3. colour intensity -> saturation (40%) + vibrance (60%)
        cur = measure(render_small(p))
        r = ref.chroma / max(cur.chroma, 1.0)
        delta = float(np.clip((r - 1.0) * 100.0 * 0.9, -45, 45))
        p.basic.saturation = float(np.clip(p.basic.saturation + delta * 0.4, -60, 60))
        p.basic.vibrance = float(np.clip(p.basic.vibrance + delta * 0.6, -60, 70))

        # ---- 4. each colour family: strength and hue
        cur = measure(render_small(p))
        for fam, (fr_r, ch_r, off_r) in ref.fam.items():
            if fam not in cur.fam:
                continue
            fr_c, ch_c, off_c = cur.fam[fam]
            if fr_c < 0.01 or fr_r < 0.01:
                continue
            m = p.mixer[fam]
            ratio = ch_r / max(ch_c, 1.0)
            m[1] = float(np.clip(m[1] + (ratio - 1.0) * 70.0, -60, 60))
            m[0] = float(np.clip(m[0] + (off_r - off_c) / 30.0 * 100.0 * 0.5, -40, 40))
        p.grading_enabled = True

    final = measure(render_small(p))
    resid = float(np.mean(np.abs(final.q - ref.q)) + np.mean([math.hypot(*(np.array(final.ab[n]) - np.array(ref.ab[n]))) for n in RANGES]) / 40.0
                  + abs(final.chroma - ref.chroma) / 60.0)
    start = float(np.mean(np.abs(first.q - ref.q)) + np.mean([math.hypot(*(np.array(first.ab[n]) - np.array(ref.ab[n]))) for n in RANGES]) / 40.0
                  + abs(first.chroma - ref.chroma) / 60.0)

    # ---- plain-language report
    def lv(q):
        return f"blacks {q[0]*100:.0f}%, mids {q[3]*100:.0f}%, highlights {q[5]*100:.0f}%"
    notes.append(f"Your photo before: {lv(first.q)}  ·  reference: {lv(ref.q)}  ·  now: {lv(final.q)}")
    if ref.q[0] > first.q[0] + 0.03:
        notes.append("Lifted the black point for the reference's matte, faded blacks.")
    elif ref.q[0] < first.q[0] - 0.03:
        notes.append("Deepened the blacks to match the reference's richer contrast.")
    if ref.q[3] > first.q[3] + 0.04:
        notes.append("Brightened the midtones.")
    elif ref.q[3] < first.q[3] - 0.04:
        notes.append("Darkened the midtones for a moodier exposure.")
    for name, label in (("shadows", "Shadows"), ("mids", "Midtones"), ("highs", "Highlights")):
        wv = getattr(p.grading, name)
        if wv.sat > 3:
            notes.append(f"{label} tinted {_hue_name(wv.hue)} (colour grading wheel, {wv.sat:.0f}).")
    sat_move = p.basic.saturation - base.basic.saturation + p.basic.vibrance - base.basic.vibrance
    if abs(sat_move) > 6:
        notes.append(("Raised" if sat_move > 0 else "Lowered") + f" colour intensity (saturation {p.basic.saturation - base.basic.saturation:+.0f}, vibrance {p.basic.vibrance - base.basic.vibrance:+.0f}).")
    shifted = [f for f in MIXER_COLORS if abs(p.mixer[f][1] - base.mixer[f][1]) > 8 or abs(p.mixer[f][0] - base.mixer[f][0]) > 8]
    if shifted:
        notes.append("Tuned the colour mixer for: " + ", ".join(shifted) + ".")
    notes.append(f"Match quality: {max(0, min(100, int(round(100 * (1 - resid / max(start, 1e-6)))))) if start > 1e-6 else 100}% closer to the reference.")
    return Match(ref_id="", base_key="", full=p, base=base, notes=notes, residual=resid)


def blend(base: EditParams, full: EditParams, k: float) -> EditParams:
    """Between your sliders (k=0) and the fully matched sliders (k=1)."""
    k = float(np.clip(k, 0.0, 1.0))
    out = full.model_copy(deep=True)
    out.basic.saturation = base.basic.saturation + (full.basic.saturation - base.basic.saturation) * k
    out.basic.vibrance = base.basic.vibrance + (full.basic.vibrance - base.basic.vibrance) * k
    bx = grade.curve_lut(base.tone.rgb)
    fx = grade.curve_lut(full.tone.rgb)
    grid = np.linspace(0, 1, 1024, dtype=np.float32)
    pts = []
    for x in CURVE_X:
        yb = float(np.interp(x, grid, bx)) if bx is not None else x
        yf = float(np.interp(x, grid, fx)) if fx is not None else x
        pts.append([x, yb + (yf - yb) * k])
    out.tone.rgb = [[round(a, 4), round(b, 4)] for a, b in pts]
    for name in RANGES:
        wb, wf, wo = getattr(base.grading, name), getattr(full.grading, name), getattr(out.grading, name)
        _set_wheel(wo, _wheel_vec(wb) + (_wheel_vec(wf) - _wheel_vec(wb)) * k, max(WHEEL_CAP[name], wb.sat))
    for f in MIXER_COLORS:
        out.mixer[f] = [base.mixer[f][i] + (full.mixer[f][i] - base.mixer[f][i]) * k for i in range(3)]
    return out
