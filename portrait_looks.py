"""Portrait looks: a small family of subtle, professional looks for photos of people.

A portrait look is not a landscape look with the sky taken out. The subject is a face: skin must stay believable (hue, brightness and colour held inside narrow
limits), the grade lives in the shadows, highlights and background, and nothing is allowed to look 'edited'. Every look is a set of SMALL deltas laid over the photo's
own Auto edit and scaled by the look strength, so 0% is the photo as it is, 100% is a refined version of it and 150% is the look pushed.

Each look is also adapted to the photo: the skin's own brightness, hue and warmth, how large the face is in the frame, whether the person is backlit, the light and
the background colour. A governor then test-renders small and eases the look back until the skin has moved no further than a retoucher would allow.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import engine
import look_engine
from params import EditParams

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


@dataclass
class PLook:
    id: str
    label: str
    blurb: str
    basic: Dict[str, float] = field(default_factory=dict)             # deltas on top of Auto, at 100%
    wheels: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)   # (hue, sat, lum) absolute, sat/lum scaled by strength
    mixer: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)    # deltas; never the skin hues
    people: Dict[str, float] = field(default_factory=dict)            # deltas on the people stage
    glow: float = 0.0
    grain: float = 0.0
    vignette: float = 0.0
    separation: float = 0.0                                            # extra background separation (dodge & burn)
    form: float = 0.0
    warm: float = 0.0                                                  # how much the look warms: limited when the skin is already warm
    keep_colour: bool = True


LOOKS: Dict[str, PLook] = {l.id: l for l in [
    PLook("clean", "Clean & Natural", "Looks like you, only better: gentle contrast, clean colour, skin kept honest",
          basic=dict(contrast=4, shadows=5, vibrance=3, clarity=2), people=dict(smooth=4, eyes=4)),
    PLook("airy", "Bright & Airy", "Light, open and fresh: lifted shadows, soft highlights, a hint of warmth",
          basic=dict(exposure=6, shadows=14, highlights=-10, contrast=-5, blacks=6, whites=2, vibrance=2),
          wheels=dict(highs=(45, 8, 2), shadows=(210, 5, 0)), people=dict(smooth=6, brighten=4, eyes=6), glow=6, warm=0.3),
    PLook("golden", "Golden Glow", "Warm, sunlit skin and a soft golden finish (never orange)",
          basic=dict(temperature=5, highlights=-8, shadows=8, vibrance=4, contrast=3),
          wheels=dict(highs=(42, 14, 2), mids=(36, 6, 0), shadows=(215, 5, 0)), people=dict(smooth=4, eyes=5, lips=2), glow=8, vignette=-6, warm=1.0),
    PLook("soft", "Soft & Dreamy", "A soft, romantic finish: gentle glow, calm contrast, smooth skin",
          basic=dict(clarity=-9, contrast=-6, highlights=-8, shadows=8, blacks=4),
          wheels=dict(highs=(48, 8, 2)), people=dict(smooth=10, eyes=5, brighten=3), glow=26, warm=0.4),
    PLook("film", "Film Portrait", "The creamy, slightly faded colour of portrait film, with a fine grain",
          basic=dict(contrast=4, blacks=10, highlights=-8, shadows=6, saturation=-4, vibrance=3),
          wheels=dict(shadows=(190, 10, 0), highs=(52, 12, 1), mids=(40, 4, 0)),
          mixer=dict(green=(-8, -14, -2), aqua=(0, -8, 0), blue=(0, -6, -4)), people=dict(smooth=3, eyes=3), grain=14, vignette=-5, warm=0.5),
    PLook("matte", "Matte Fashion", "A lifted matte black with calm, fashion-magazine colour",
          basic=dict(blacks=14, highlights=-14, contrast=4, saturation=-8, vibrance=-2, clarity=2),
          wheels=dict(shadows=(228, 8, 0), highs=(34, 6, 0)), mixer=dict(green=(0, -16, -4), blue=(0, -10, -4)), people=dict(smooth=4, eyes=4), warm=0.2),
    PLook("editorial", "Editorial Punch", "Crisp, confident contrast with a subtle vignette, the magazine-cover look",
          basic=dict(contrast=14, blacks=-8, whites=5, clarity=5, vibrance=6, highlights=-6),
          wheels=dict(shadows=(215, 8, 0), highs=(40, 6, 0)), people=dict(smooth=3, eyes=8, lips=3, sculpt=6), vignette=-12, separation=8, form=10, warm=0.2),
    PLook("moody", "Moody Low-Key", "Darker, atmospheric surroundings with the face kept lit",
          basic=dict(exposure=-4, contrast=12, blacks=-10, shadows=-4, highlights=-12, saturation=-6, clarity=3),
          wheels=dict(shadows=(212, 12, -1), highs=(40, 8, 0)), people=dict(eyes=7, lift=14, sculpt=8), vignette=-22, separation=22, form=14, warm=0.2),
    PLook("cine", "Cinematic Teal", "Teal shadows and surroundings with warm skin: the cinema grade, restrained",
          basic=dict(contrast=8, highlights=-10, shadows=4, blacks=-4, vibrance=2),
          wheels=dict(shadows=(196, 16, -1), highs=(40, 12, 1), mids=(38, 4, 0)), mixer=dict(blue=(-4, 8, -8), aqua=(0, 12, -4), green=(-8, -6, -4)),
          people=dict(eyes=5, sculpt=4), vignette=-8, warm=0.6),
    PLook("bw", "Classic Black & White", "Timeless monochrome with rich blacks and luminous skin",
          basic=dict(saturation=-100, contrast=12, blacks=-6, whites=4, clarity=6, highlights=-8, shadows=6),
          people=dict(smooth=4, eyes=6, sculpt=5), vignette=-10, grain=12, keep_colour=False),
]}


@dataclass
class PScene:
    people: int = 0
    face_frac: float = 0.0           # share of the picture the largest face covers
    person_frac: float = 0.0
    skin_L: float = 55.0             # median Lab lightness of the skin
    skin_hue: float = 50.0           # degrees, Lab hue of the skin (about 35-65 is the natural range)
    skin_chroma: float = 20.0
    backlit: float = 0.0             # 0..1
    light: str = "sunny"
    sky_blue: float = 0.0
    green: float = 0.0
    bg_busy: float = 0.5
    median: float = 0.45
    close: bool = False


def read_scene(an) -> PScene:
    cached = getattr(an, "_portrait_scene", None)
    if cached is not None:
        return cached
    s = PScene(light=an.read.light if an.read else "sunny", median=float(an.stats.get("median", 0.45)))
    faces = [f for f in (an.summary().get("faces") or [])]
    s.people = len(faces)
    s.face_frac = max([f.get("size_pct", 0.0) for f in faces] or [0.0]) / 100.0
    s.person_frac = float(an.stats.get("person_frac", 0.0))
    s.close = s.face_frac > 0.02 or s.person_frac > 0.35
    src, size, _ = engine._target(an, 480)
    sk = an.bank.get("skin", (size[0], size[1])) if an.bank is not None else None
    if sk is not None and (sk > 0.6).sum() > 150:
        lab = cv2.cvtColor(np.clip(src, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
        sel = sk > 0.6
        s.skin_L = float(np.median(lab[..., 0][sel]))
        a, b = float(np.median(lab[..., 1][sel])), float(np.median(lab[..., 2][sel]))
        s.skin_hue, s.skin_chroma = float(np.degrees(np.arctan2(b, a)) % 360), float(np.hypot(a, b))
    pl, bg = an.stats.get("person_luma"), None
    pm = an.bank.get("person", (size[0], size[1])) if an.bank is not None else None
    y = (np.clip(src, 0, 1) @ LUMA)
    if pm is not None and (pm < 0.1).sum() > 200:
        bg = float(np.median(y[pm < 0.1]))
    if pl is not None and bg is not None:
        s.backlit = float(np.clip((bg - pl - 0.12) / 0.30, 0, 1))
    try:
        import drama_looks
        d = drama_looks.read_scene(an)
        s.sky_blue, s.green, s.light = float(d.sky_blue), float(d.green), d.light
    except Exception:
        pass
    br = an.stats.get("bg_detail_ratio")
    s.bg_busy = float(np.clip((br if br is not None else 0.8) / 1.5, 0, 1))
    try:
        an._portrait_scene = s
    except Exception:
        pass
    return s


def score(l: PLook, s: PScene) -> Tuple[float, str]:
    sc, why = 0.40, ""

    def add(x, reason):
        nonlocal sc, why
        sc += x
        if x >= 0.1 and not why:
            why = reason
    lt = s.light
    if l.id == "clean":
        add(0.16, "a safe, honest finish that suits any photo")
        if s.people >= 3: add(0.18, "a group: keep every face natural")
    elif l.id == "airy":
        if lt in ("overcast", "indoor") or s.median < 0.42 and s.backlit < 0.3: add(0.22, "flat or dim light that gets lighter and fresher")
        if s.skin_L < 52 and s.person_frac > 0.1: add(0.12, "brightens a face that sits a little dark")
        if lt in ("sunset", "golden_hour"): sc -= 0.08
    elif l.id == "golden":
        if lt in ("overcast", "blue_hour"): add(0.24, "warm light that flatters skin under a grey sky")
        if lt in ("golden_hour", "sunset"): add(0.20, "already warm light, enhanced gently")
        if s.skin_hue > 62 or s.skin_chroma > 30: sc -= 0.22        # skin already warm/orange: never push it further
    elif l.id == "soft":
        if s.close: add(0.18, "a close portrait that suits a soft, glowing finish")
        if lt == "sunny" and s.face_frac > 0.03: add(0.08, "softens harsh daylight on a face")
    elif l.id == "film":
        if s.green > 0.25: add(0.20, "greens that film colour tames and warms")
        if lt in ("sunny", "golden_hour", "general"): add(0.08, "daylight colour with room for a creamy tone")
    elif l.id == "matte":
        if s.green > 0.2 or s.sky_blue > 0.2: add(0.14, "bright, saturated surroundings that a matte finish calms")
        if lt in ("sunny",): add(0.06, "contrasty daylight softened into a matte")
    elif l.id == "editorial":
        if s.close and s.skin_L > 45: add(0.20, "a close, well-lit face that carries crisp contrast")
        if lt in ("indoor", "studio"): add(0.08, "controlled light")
    elif l.id == "moody":
        if s.bg_busy > 0.5 and s.skin_L > 40: add(0.12, "a busy background that can fall back into shadow")
        if s.backlit > 0.5: sc -= 0.1
        if s.skin_L < 38: sc -= 0.28                                  # a darker face must not be made darker
        if lt in ("blue_hour", "night", "indoor"): add(0.14, "low light that already has a mood")
    elif l.id == "cine":
        if s.green > 0.25 or s.sky_blue > 0.25: add(0.16, "green and blue surroundings that turn teal while skin stays warm")
        if s.skin_hue < 30: sc -= 0.12
    elif l.id == "bw":
        sc -= 0.06
        if lt in ("overcast", "indoor") and s.green < 0.2 and s.sky_blue < 0.1: add(0.18, "muted colour: tone and expression do the work")
        if s.skin_chroma < 14: add(0.06, "little colour to lose")
    return float(np.clip(sc, 0.05, 1.0)), why or "a good starting point"


def generate(an) -> List[dict]:
    s = read_scene(an)
    out = []
    for l in LOOKS.values():
        f, why = score(l, s)
        out.append({"id": l.id, "label": l.label, "blurb": l.blurb, "fit": round(f, 2), "reason": why})
    out.sort(key=lambda d: -d["fit"])
    for i, d in enumerate(out):
        d["recommended"] = i < 3
    return out


def _skin_stats(an, p: EditParams, size=480):
    src, sz, _ = engine._target(an, size)
    sk = an.bank.get("skin", (sz[0], sz[1])) if an.bank is not None else None
    if sk is None or (sk > 0.6).sum() < 150:
        return None, None
    out = np.asarray(engine.render(an, p, size), np.float32) / 255
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2Lab)
    sel = sk > 0.6
    a, b = float(np.median(lab[..., 1][sel])), float(np.median(lab[..., 2][sel]))
    return (float(np.median(lab[..., 0][sel])), a, b), out


def build(an, base: EditParams, lid: str, strength: float = 1.0) -> Tuple[EditParams, List[str], dict]:
    """The photo with portrait look `lid` at `strength` (0..1.5) on top of `base` (its Auto / current edit)."""
    l = LOOKS[lid]
    s = read_scene(an)
    k = float(np.clip(strength, 0.0, 1.5))
    p = base.model_copy(deep=True)
    notes: List[str] = [f"{l.label}: {l.blurb}"]

    # ---- how hard this photo can take the look
    kk = k * 1.45                                                    # the library numbers are deliberately small; this sets the 100% strength
    if s.people >= 3:
        kk *= 0.8                                                      # a group: every face has to survive
    warm_room = float(np.clip((68.0 - s.skin_hue) / 22.0, 0.0, 1.0)) * float(np.clip((36.0 - s.skin_chroma) / 14.0, 0.2, 1.0))   # little room when skin is already orange
    wk = kk * (1.0 - l.warm + l.warm * warm_room)
    if l.warm and warm_room < 0.6:
        notes.append("Skin is already warm → the warm push is held back")

    b = p.basic
    for key, v in l.basic.items():
        d = v * (wk if key in ("temperature", "tint") else kk)
        if key == "exposure" and v < 0 and s.skin_L < 45:
            d *= 0.3                                                  # a face that sits dark is not darkened further
        if key == "shadows" and s.backlit > 0.4:
            d *= 1.3
        setattr(b, key, float(np.clip(getattr(b, key) + d, -100, 100)))
    if not l.keep_colour:
        b.saturation, b.vibrance = -100.0, 0.0
        p.mixer = {c: [0.0, 0.0, 0.0] for c in p.mixer}

    if l.wheels:
        p.grading_enabled = True
        for name, (h, sat, lum) in l.wheels.items():
            w = getattr(p.grading, name)
            sat_k = sat * (wk if 0 <= h <= 70 else kk)
            if w.sat < 4 or sat_k > w.sat * 0.6:
                w.hue, w.sat = float(h), float(np.clip(max(w.sat * 0.5, sat_k), 0, 100))
            w.lum = float(np.clip(w.lum + lum * kk, -100, 100))
        p.grading.blending = max(p.grading.blending, 60.0)
    for color, (h, sa, lu) in l.mixer.items():
        m = p.mixer.get(color, [0.0, 0.0, 0.0])
        p.mixer[color] = [m[0] + h * kk, m[1] + sa * kk, m[2] + lu * kk]

    pe = p.people
    if "person" in an.bank.names():
        pe.enabled = True
        for key, v in l.people.items():
            if key == "smooth":
                v *= float(np.clip(0.4 + 6 * s.face_frac, 0.4, 1.0))    # small faces need less smoothing: it would only blur them
            setattr(pe, key, float(np.clip(getattr(pe, key) + v * min(kk, 1.3 * k + 0.1), 0, 100)))
        pe.tone_fix = max(pe.tone_fix, 60.0)
        if s.skin_L < 40:                                                # a dark face gets a gentle lift, never a darker look
            pe.lift = max(pe.lift, 10.0 * kk)
    e = p.effects
    e.glow = float(np.clip(e.glow + l.glow * kk, 0, 100))
    e.grain = float(np.clip(e.grain + l.grain * 0.45 * k, 0, 100))
    e.vignette = float(np.clip(e.vignette + l.vignette * kk * (0.6 if s.people >= 3 else 1.0), -100, 100))
    db = p.dodge_burn
    db.separation = float(np.clip(db.separation + l.separation * kk, 0, 100))
    db.form = float(np.clip(db.form + l.form * kk * float(np.clip(0.5 + 8 * s.face_frac, 0.5, 1.0)), 0, 100))
    if l.separation >= 15 and s.bg_busy < 0.3:
        db.separation = base.dodge_burn.separation                  # the background is already soft: do not darken it twice

    # ---- governor: the face must stay a believable face
    if l.keep_colour and k > 0:
        st0, _ = _skin_stats(an, base)
        if st0 is not None:
            for _ in range(4):
                st1, out = _skin_stats(an, p)
                if st1 is None:
                    break
                c0, c1 = float(np.hypot(st0[1], st0[2])), float(np.hypot(st1[1], st1[2]))
                h0, h1 = np.degrees(np.arctan2(st0[2], st0[1])), np.degrees(np.arctan2(st1[2], st1[1]))
                dh = abs((h1 - h0 + 180) % 360 - 180)
                dl = st1[0] - st0[0]
                if c1 <= c0 * 1.14 + 1.0 and c1 >= c0 * 0.78 and dh <= 6.0 and abs(dl) <= (10.0 if lid in ("moody", "airy") else 7.0):
                    break
                f = 0.78
                for key in l.basic:
                    setattr(p.basic, key, getattr(base.basic, key) + (getattr(p.basic, key) - getattr(base.basic, key)) * f)
                for name in l.wheels:
                    w, w0 = getattr(p.grading, name), getattr(base.grading, name)
                    w.sat = w0.sat + (w.sat - w0.sat) * f
                p.effects.glow = base.effects.glow + (p.effects.glow - base.effects.glow) * f
                notes.append("Eased back so the skin stays natural")
    notes += look_engine._govern_exposure(an, p, base)
    p.look = base.look
    return p, notes, {"portrait": lid, "skin_L": round(s.skin_L, 1), "skin_hue": round(s.skin_hue, 1), "faces": s.people}
