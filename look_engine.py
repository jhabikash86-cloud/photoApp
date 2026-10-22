"""Scene-aware look engine: one click gives a finished, professional edit that is built from what is IN the photo.

A look is not a filter. For each photo it combines three layers, all written as ordinary editable sliders:

  1. COLOUR GRADING  the look's global grade (tone, grading wheels, colour mixer, curve), adapted to the light and scene the
                     analysis read (a golden-hour photo does not get warmed twice, an overcast one is not pushed toward
                     gold, haze raises dehaze, snow keeps its whites, indoor scenes get no landscape clarity).
  2. SELECTIVE MASKS every region found (sky, mountains, water, trees, grass, buildings, ground, foreground, miscellaneous)
                     gets its own adjustment, shaped by the look's character and by that region's size and mask accuracy;
                     skin is protected.
  3. LIGHTING        the lighting analysis (dodge & burn proposal, subject / background separation) plus a foreground lift,
                     scaled by the look.

A governor then test-renders the result small and backs off whatever went too far (exposure, clipped highlights, crushed
shadows, over-saturation, skin getting more colourful), so a look never ruins a picture.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import engine
import scene_grade
from params import EditParams, MIXER_COLORS

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
ADJ = ("exposure", "contrast", "highlights", "shadows", "vibrance", "saturation", "clarity", "dehaze", "temperature", "tint", "hue")
MIN_REGION_PCT = 1.5            # regions smaller than this share of the picture are left alone (their mask is too small to be worth it)


@dataclass
class Feel:
    """Character of a look, each -1..1 (0 = neutral)."""
    warm: float = 0.0       # warm light / warm foliage
    depth: float = 0.0      # darker, richer sky / water / trees
    punch: float = 0.0      # local contrast and texture
    colour: float = 0.0     # colour richness (negative = muted)
    lift: float = 0.0       # open up dark foreground detail


@dataclass
class Look:
    id: str
    label: str
    result: str
    adjusts: str
    feel: Feel
    basic: Dict[str, float]
    wheels: Dict[str, tuple]                              # hue, saturation, luminance
    blending: float = 55.0
    balance: float = 0.0
    mixer: Dict[str, tuple] = field(default_factory=dict)
    curve: Dict[str, float] = field(default_factory=dict)
    vignette: float = 0.0


LOOKS: Dict[str, Look] = {l.id: l for l in [
    Look("cinematic", "Cinematic Landscape", "Dramatic, premium look", "Teal shadows, warm highlights, controlled contrast",
         Feel(0.3, 0.6, 0.7, 0.5, 0.4),
         dict(contrast=16, highlights=-18, shadows=8, whites=4, blacks=-14, clarity=8, dehaze=6, vibrance=12),
         dict(shadows=(205, 32, -2), mids=(30, 8, 0), highs=(38, 30, 2)), 60, 0,
         dict(orange=(0, 10, 6), aqua=(0, 16, -8), blue=(-4, 12, -12), green=(-10, -8, -8)), dict(shadows=-5, highlights=4), -10),
    Look("golden", "Golden Hour", "Warm, glowing scenery", "Warm highlights, golden vegetation, balanced shadows",
         Feel(1.0, 0.3, 0.4, 0.7, 0.5),
         dict(contrast=12, highlights=-22, shadows=14, whites=6, blacks=-10, clarity=6, vibrance=18, temperature=10),
         dict(highs=(40, 36, 3), mids=(32, 12, 0), shadows=(215, 10, 0)), 60, 6,
         dict(yellow=(-8, 10, 0), green=(-14, -4, -6), orange=(0, 14, 6)), {}, 0),
    Look("alpine", "Alpine & Mountain", "Crisp, dimensional mountains", "Cool shadows, detailed rocks, natural snow",
         Feel(-0.4, 0.4, 1.0, 0.4, 0.5),
         dict(contrast=14, highlights=-14, shadows=10, whites=8, blacks=-10, clarity=18, texture=12, dehaze=12, vibrance=10, temperature=-6),
         dict(shadows=(215, 24, 0), highs=(50, 8, 0)), 55, 0,
         dict(blue=(-2, 12, -12), aqua=(0, 8, -4), green=(-6, -6, -6)), {}, 0),
    Look("tropical", "Tropical Paradise", "Vibrant but natural", "Aqua water, rich greens, controlled blues",
         Feel(0.1, 0.3, 0.3, 1.0, 0.5),
         dict(contrast=8, highlights=-10, shadows=10, clarity=6, vibrance=22, saturation=2, temperature=4),
         dict(highs=(50, 12, 0), shadows=(190, 14, 0)), 55, 0,
         dict(aqua=(0, 26, -6), green=(4, 18, -6), blue=(4, -6, -10), yellow=(0, 6, 0)), {}, 0),
    Look("moody", "Moody Landscape", "Dark, atmospheric scenery", "Deeper shadows, muted colours, selective highlights",
         Feel(-0.2, 1.0, 0.7, -0.4, 0.0),
         dict(exposure=-6, contrast=18, highlights=-22, shadows=-4, blacks=-16, saturation=-12, vibrance=4, clarity=12, dehaze=8),
         dict(shadows=(215, 20, 0), mids=(200, 6, 0), highs=(45, 10, 0)), 55, 0,
         dict(green=(0, -10, -10), blue=(0, -6, -12), orange=(0, 4, 0)), {}, -18),
    Look("natural", "Natural Professional", "Clean, realistic photography", "Balanced exposure, subtle colour grading",
         Feel(0.1, 0.2, 0.3, 0.4, 0.4),
         dict(contrast=8, highlights=-12, shadows=10, whites=4, blacks=-6, clarity=6, dehaze=4, vibrance=12),
         dict(highs=(45, 10, 0), shadows=(215, 10, 0)), 55, 0, {}, {}, 0),
    Look("fine_art", "Fine Art Landscape", "Soft, sophisticated finish", "Restrained saturation, tonal separation",
         Feel(0.2, 0.3, -0.2, -0.5, 0.5),
         dict(contrast=6, highlights=-14, shadows=10, blacks=4, saturation=-14, vibrance=6, clarity=-4, texture=-4),
         dict(shadows=(200, 16, 0), mids=(35, 6, 0), highs=(40, 14, 0)), 70, 0, {}, dict(shadows=6, highlights=-4), -8),
    Look("sunset", "Sunset & Sunrise", "Dramatic skies", "Warm highlights, orange and magenta refinement",
         Feel(0.9, 0.7, 0.6, 0.8, 0.4),
         dict(contrast=14, highlights=-30, shadows=14, blacks=-12, dehaze=6, clarity=8, vibrance=18, temperature=6),
         dict(highs=(22, 42, 2), mids=(340, 12, 0), shadows=(255, 24, -2)), 65, 10,
         dict(orange=(0, 18, 4), red=(0, 10, 0), magenta=(0, 14, 0), purple=(0, 12, -6), yellow=(-4, 6, 0)), {}, 0),
]}


def list_looks() -> List[dict]:
    return [{"id": l.id, "label": l.label, "result": l.result, "adjusts": l.adjusts} for l in LOOKS.values()]


# ------------------------------------------------------------------------------------------------ scene facts
@dataclass
class Facts:
    kind: str
    light: str
    scene: str
    sky: float = 0.0               # share of the picture, 0..1
    haze: float = 0.0
    overcast: float = 0.0          # 0 clear colourful sky .. 1 pale grey
    snowy: bool = False
    indoor: bool = False
    has_person: bool = False
    present: Dict[str, float] = field(default_factory=dict)


def read_facts(an, light: Optional[str], scene: Optional[str]) -> Facts:
    r = an.read
    lt = light if light in scene_grade.LIGHTS else (r.light if r else "sunny")
    sc = scene if scene in scene_grade.SCENES else (r.scene if r else "general")
    names = an.bank.names()
    cov = an.coverage or {}
    present = {}
    for n in names:
        if n in ("person", "skin", "background"):
            continue
        present[n] = cov.get(n, 12.0 if n == "foreground" else 0.0)
    size = (an.work.shape[1], an.work.shape[0])
    snowy = sc == "snow"
    mt = an.bank.get("mountains", size) if "mountains" in names else None
    if mt is not None and (mt > 0.5).mean() > 0.01:
        sel = mt > 0.5
        y = (an.work @ LUMA)[sel]
        hsv = cv2.cvtColor(an.work, cv2.COLOR_RGB2HSV)
        snowy = snowy or (float(np.median(y)) > 0.66 and float(np.median(hsv[..., 1][sel])) < 0.18)
    sat_sky = an.stats.get("sky_sat", 0.3)
    return Facts(kind=an.scene.kind, light=lt, scene=sc, sky=cov.get("sky", 0.0) / 100.0, haze=float(an.scene.haze),
                 overcast=float(np.clip((0.22 - sat_sky) / 0.15, 0, 1)), snowy=snowy, indoor=bool(an.sem_summary.get("indoor")),
                 has_person="person" in an.masks, present=present)


# ------------------------------------------------------------------------------------------------ adaptive: read the tones
@dataclass
class Tonal:
    """How this particular photo is exposed. Two mountain photos can need opposite treatment; these numbers say which."""
    p5: float = 0.05
    p50: float = 0.45
    p95: float = 0.9
    rng: float = 0.8              # p95 - p5: the tonal range actually used
    clip_hi: float = 0.0          # share of the picture that is nearly white
    crush_lo: float = 0.0         # share that is nearly black
    sky_med: Optional[float] = None
    sky_clip: float = 0.0         # share of the SKY that is nearly white
    fg_med: Optional[float] = None
    mt_med: Optional[float] = None
    mt_range: Optional[float] = None
    trees_med: Optional[float] = None
    # derived 0..1 "how much is this the case"
    gap: float = 0.0              # sky brighter than the foreground
    flat: float = 0.0             # flat, low-contrast light (overcast / hazy / evenly lit)
    contrasty: float = 0.0        # already a very wide range
    fg_dark: float = 0.0
    fg_bright: float = 0.0
    mt_flat: float = 0.0          # mountains evenly lit: little tonal separation
    trees_dark: float = 0.0


def _c(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def measure_tonal(an) -> Tonal:
    y = an.work @ LUMA
    p5, p50, p95 = (float(v) for v in np.percentile(y, [5, 50, 95]))
    t = Tonal(p5=p5, p50=p50, p95=p95, rng=p95 - p5, clip_hi=float((y > 0.97).mean()), crush_lo=float((y < 0.03).mean()))
    size = (an.work.shape[1], an.work.shape[0])
    names = an.bank.names()

    def stats(name, min_share=0.01):
        if name not in names:
            return None
        m = an.bank.get(name, size)
        if m is None or (m > 0.5).mean() < min_share:
            return None
        sel = m > 0.5
        v = y[sel]
        return float(np.median(v)), float(np.percentile(v, 90) - np.percentile(v, 10)), float((v > 0.97).mean())

    if (s := stats("sky")):
        t.sky_med, t.sky_clip = s[0], s[2]
    if (s := stats("foreground", 0.02)):
        t.fg_med = s[0]
    if (s := stats("mountains")):
        t.mt_med, t.mt_range = s[0], s[1]
    if (s := stats("trees")):
        t.trees_med = s[0]
    if t.sky_med is not None and t.fg_med is not None:
        t.gap = _c((t.sky_med - t.fg_med - 0.12) / 0.35)
    t.flat = _c((0.62 - t.rng) / 0.30)
    t.contrasty = _c((t.rng - 0.82) / 0.14)
    if t.fg_med is not None:
        t.fg_dark = _c((0.30 - t.fg_med) / 0.15)
        t.fg_bright = _c((t.fg_med - 0.52) / 0.20)
    if t.mt_range is not None:
        t.mt_flat = _c((0.22 - t.mt_range) / 0.12)
    if t.trees_med is not None:
        t.trees_dark = _c((0.22 - t.trees_med) / 0.10)
    return t


def adaptive_strength(t: Tonal) -> float:
    """Flat light and big sky/foreground gaps can carry more of a look; an already contrasty picture needs less."""
    return float(min(1.2, max(0.8, 1.0 + 0.12 * t.flat + 0.10 * t.gap - 0.15 * t.contrasty)))


def global_multipliers(t: Tonal) -> Dict[str, float]:
    clip_g = _c(t.clip_hi / 0.05)
    return {
        "contrast": 1 + 0.7 * t.flat - 0.6 * t.contrasty,
        "clarity": 1 + 0.5 * t.flat - 0.4 * t.contrasty,
        "dehaze": 1 + 0.6 * t.flat,
        "highlights": 1 + 1.0 * clip_g + 0.5 * t.gap - 0.5 * t.flat,
        "shadows": 1 + 0.8 * t.fg_dark + 0.5 * t.gap - 0.3 * t.flat + 0.6 * _c(t.crush_lo / 0.05),
        "whites": 1 - 0.8 * clip_g,
    }


def adapt_regions(plan: Dict[str, Dict[str, float]], t: Tonal) -> Dict[str, Dict[str, float]]:
    """Scale each region's adjustments by how its own tones look in this photo."""
    sky_clip = _c(t.sky_clip * 10)
    mult = {
        "sky": dict(highlights=0.6 + 1.1 * t.gap + sky_clip, exposure=(0.5 + 1.2 * t.gap) * (1 - 0.5 * t.flat),
                    contrast=1 + 0.9 * t.flat, clarity=1 + 0.9 * t.flat, vibrance=1 - 0.3 * t.flat),
        "foreground": dict(shadows=0.6 + 0.9 * t.gap + 0.8 * t.fg_dark, exposure=(0.4 + 1.0 * t.fg_dark) * (1 - t.fg_bright),
                           clarity=1 + 0.4 * t.flat),
        "mountains": dict(clarity=1 + 0.9 * t.mt_flat, contrast=1 + 0.9 * t.mt_flat, highlights=1 + 0.5 * t.gap),
        "trees": dict(exposure=1 - 0.7 * t.trees_dark, shadows=1 - 0.7 * t.trees_dark),
        "water": dict(highlights=1 + 0.5 * t.gap),
    }
    for name, ms in mult.items():
        if name in plan:
            for key, m in ms.items():
                if key in plan[name]:
                    plan[name][key] *= max(0.0, m)
    return plan


def adaptive_notes(t: Tonal, k_adapt: float) -> List[str]:
    n: List[str] = []
    if t.gap > 0.45:
        n.append(f"Bright sky over a darker foreground (gap {t.sky_med - t.fg_med:.2f}) → sky burned harder, foreground lifted more")
    if t.sky_clip > 0.02:
        n.append(f"{t.sky_clip * 100:.0f}% of the sky is near-white → extra highlight recovery there")
    if t.flat > 0.3:
        n.append(f"Flat light (tonal range {t.rng:.2f}) → more contrast and clarity, less sky darkening")
    if t.contrasty > 0.3:
        n.append(f"Already a wide tonal range ({t.rng:.2f}) → contrast and clarity held back")
    if t.mt_flat > 0.3:
        n.append("Evenly lit mountains → extra tonal separation")
    if t.fg_dark > 0.4:
        n.append(f"Dark foreground (brightness {t.fg_med:.2f}) → shadows opened")
    if t.trees_dark > 0.4:
        n.append("Trees already dark → not deepened further")
    if abs(k_adapt - 1) >= 0.03:
        n.append(f"Adaptive strength ×{k_adapt:.2f} for this photo's light")
    return n


# ------------------------------------------------------------------------------------------------ the three layers
def _region_plan(f: Feel, facts: Facts) -> Dict[str, Dict[str, float]]:
    """Per-region adjustments at full strength, from the look's character and what the scene is like."""
    w, d, p, c, lift = f.warm, f.depth, f.punch, f.colour, f.lift
    soften = 1 - 0.45 * facts.overcast                      # pale grey skies take colour badly: be gentler there
    plan: Dict[str, Dict[str, float]] = {
        # refine blues, control highlights, enhance cloud contrast
        "sky": dict(exposure=-(3 + 7 * d), highlights=-(8 + 14 * d), contrast=(4 + 10 * p) * (1 + 0.3 * min(1, facts.sky / 0.4)),
                    clarity=4 + 14 * p, dehaze=(4 + 10 * p) * soften, vibrance=(6 + 10 * c) * soften, saturation=min(0, 8 * c),
                    temperature=12 * w - 2),
        # tonal separation, snow detail kept
        "mountains": dict(exposure=2 - 2 * d, contrast=6 + 10 * p, clarity=8 + 14 * p, highlights=-(10 + 6 * d), shadows=6 + 8 * lift,
                          dehaze=4 + 8 * p, vibrance=4 + 6 * c, temperature=14 * w),
        # enhance blue / aqua, keep reflections
        "water": dict(exposure=-(2 + 6 * d), highlights=-(8 + 8 * d), contrast=4 + 6 * p, clarity=2 + 4 * p, vibrance=10 + 14 * c,
                      saturation=2 + 4 * c, temperature=-6 + 12 * w),
        # refine greens, control saturation, deepen shadows
        "trees": dict(exposure=-(3 + 7 * d), shadows=-(4 + 8 * d), saturation=float(np.clip(-4 + 4 * c, -8, 2)), vibrance=6 + 10 * c,
                      contrast=4 + 6 * p, clarity=4 + 6 * p, temperature=8 * w),
        "grass": dict(exposure=-(1 + 3 * d), saturation=float(np.clip(-4 + 4 * c, -8, 2)), vibrance=6 + 12 * c, contrast=3 + 4 * p, temperature=12 * w),
        # lift dark details, subtle local contrast
        "foreground": dict(exposure=1 + 4 * lift, shadows=8 + 14 * lift, clarity=6 + 8 * p, contrast=3 + 5 * p, vibrance=4 + 6 * c, temperature=6 * w),
        "ground": dict(contrast=4 + 6 * p, clarity=6 + 8 * p, shadows=6 + 8 * lift, temperature=10 * w, vibrance=4 + 6 * c),
        "buildings": dict(clarity=6 + 10 * p, contrast=4 + 6 * p, temperature=6 * w, vibrance=2 + 4 * c),
        "other": dict(contrast=3, shadows=4 + 4 * lift),
    }
    if facts.snowy:                                         # white snow must keep its texture: never push it
        plan["mountains"]["highlights"] -= 8
        plan["mountains"]["exposure"] = min(plan["mountains"]["exposure"], 0)
        plan["mountains"]["vibrance"] *= 0.5
    if facts.indoor:
        for r in plan.values():
            r["clarity"] = r.get("clarity", 0) * 0.5
            r["dehaze"] = r.get("dehaze", 0) * 0.3
    return plan


def _clip(obj, name, lo, hi):
    setattr(obj, name, float(min(hi, max(lo, getattr(obj, name)))))


def build(an, base: EditParams, look: Look, strength: float, light: Optional[str] = None, scene: Optional[str] = None) -> Tuple[EditParams, List[str], dict]:
    """The photo with `look` applied at `strength` (0..1.5). Returns (params, plain-language notes, facts for the UI)."""
    k = float(np.clip(strength, 0.0, 1.5))
    facts = read_facts(an, light, scene)
    tonal = measure_tonal(an)
    k_adapt = adaptive_strength(tonal)
    k = float(np.clip(k * k_adapt, 0.0, 1.8))         # the slider is the user's; the photo decides how much a look can carry
    gmult = global_multipliers(tonal)
    notes: List[str] = []
    p = base.model_copy(deep=True)
    neutral = EditParams()
    wb_t, wb_ti = p.basic.temperature, p.basic.tint

    # ---- start from a clean tone / colour slate; white balance, people, masks and finishing effects are the user's
    p.basic = neutral.basic.model_copy()
    p.basic.temperature, p.basic.tint = wb_t, wb_ti
    p.mixer = {c: [0.0, 0.0, 0.0] for c in MIXER_COLORS}
    p.grading = neutral.grading.model_copy(deep=True)
    p.curve = neutral.curve.model_copy()
    p.grading_enabled = True
    for n, r in p.regions.items():
        for a in ADJ:
            setattr(r, a, 0.0)
    where = []

    # ---- 1. colour grading, adapted to the light and scene
    warm_atten = 1.0
    if facts.light in ("golden_hour", "sunset"):
        warm_atten = 0.55           # the light is already warm: do not warm it twice
        notes.append(f"Light already warm ({scene_grade.LIGHT_LABEL.get(facts.light, facts.light)}) → warm push reduced")
    elif facts.light in ("overcast", "blue_hour", "night", "indoor") and look.feel.warm > 0.5:
        warm_atten = 0.75
        notes.append(f"{scene_grade.LIGHT_LABEL.get(facts.light, facts.light)} light → gold toned down so it stays believable")
    b = p.basic
    for key, v in look.basic.items():
        if key == "temperature" and v > 0:
            v *= warm_atten
        v *= max(0.0, gmult.get(key, 1.0))
        setattr(b, key, getattr(b, key) + v * k)
    haze_boost = min(14.0, 18 * facts.haze)
    if haze_boost > 3:
        b.dehaze += haze_boost * k
        notes.append(f"Haze detected ({facts.haze:.2f}) → dehaze +{haze_boost * k:.0f}")
    if facts.snowy:
        b.whites = min(b.whites, 0.0)
        b.highlights -= 6 * k
        b.clarity *= 0.8
        notes.append("Snow / bright rock → whites held back, highlights protected")
    if facts.indoor:
        b.clarity *= 0.5
        b.dehaze *= 0.3
        notes.append("Indoor scene → landscape clarity and dehaze reduced")
    if an.scene.kind in ("portrait", "people"):
        b.clarity *= 0.5
        b.texture *= 0.3
        notes.append("People in the frame → clarity and texture softened for natural skin")
    for name, (h, s_, l) in look.wheels.items():
        wh = getattr(p.grading, name)
        sat = s_ * k * (warm_atten if name in ("highs", "mids") and 0 <= h <= 70 else 1.0)
        wh.hue, wh.sat, wh.lum = float(h), float(min(100, sat)), l * k
    p.grading.blending, p.grading.balance = look.blending, look.balance * k
    for color, (h, s_, l) in look.mixer.items():
        p.mixer[color] = [h * k, s_ * k, l * k]
    for key, v in look.curve.items():
        setattr(p.curve, key, v * k)
    p.effects.vignette = look.vignette * k if an.scene.kind not in ("portrait",) else min(look.vignette * k, 0) * 0.5
    notes.append(f"{look.label}: {look.adjusts.lower()}")
    notes += adaptive_notes(tonal, k_adapt)

    # ---- 2. selective masks, one adjustment per detected region
    plan = adapt_regions(_region_plan(look.feel, facts), tonal)
    for name, adj in plan.items():
        if name not in p.regions or name not in facts.present:
            continue
        share = facts.present[name]
        if share < MIN_REGION_PCT and name != "foreground":
            continue
        acc = an.reports[name].strength if name in an.reports else 1.0      # a less accurate mask is edited more gently
        r = p.regions[name]
        r.enabled = True
        for a, v in adj.items():
            setattr(r, a, v * k * acc)
        where.append(name)
    labels = {"sky": "Sky", "mountains": "Mountains", "water": "Water", "trees": "Trees", "grass": "Grass", "foreground": "Foreground",
              "ground": "Ground", "buildings": "Buildings", "other": "Miscellaneous"}
    if where:
        notes.append("Region adjustments: " + ", ".join(labels.get(n, n) for n in where))

    # ---- 3. people: protect skin and keep it natural
    if "skin" in p.regions and "skin" in an.bank.names():
        r = p.regions["skin"]
        r.enabled = True
        r.saturation = -(3 + 4 * max(look.feel.colour, 0)) * k
        r.temperature = -0.7 * max(b.temperature, 0) - 4 * max(look.feel.warm, 0) * k
        r.highlights = -4 * k
        if p.people.enabled:
            p.people.tone_fix = max(p.people.tone_fix, 60)
        notes.append("People detected → skin protected (colour held back, tone kept natural)")

    # ---- 4. intelligent lighting
    db = engine.auto_lighting(an)
    scale = k * (0.6 + 0.4 * look.feel.depth)
    for fld in ("dodge", "burn", "separation", "form"):
        setattr(db, fld, float(getattr(db, fld)) * scale)
    p.dodge_burn = db       # (blended with the current lighting below, by strength)
    notes += list(an.lighting_notes[:3])
    if "sky" in where and plan["sky"]["highlights"] < -12:
        notes.append("Bright sky → burned gently with the sky mask so clouds keep detail")
    if "foreground" in where:
        notes.append("Foreground → dark detail lifted, subtle local contrast added")

    # ---- strength: 0 = your edit exactly as it is, 100 = the full look. Below 100 the rest of your current edit is kept.
    _keep_current(p, base, where, min(k / k_adapt, 1.0))

    # ---- 5. governor: test-render small and back off anything that went too far
    p = engine._govern(an, p)
    engine._limit_tone(p)
    notes += _govern_exposure(an, p, base)
    for fld in ("temperature",):
        _clip(p.basic, fld, -100, 100)
    p.look = base.look
    p.lut = base.lut
    return p, notes, {"light": facts.light, "scene": facts.scene, "regions": where, "snowy": facts.snowy, "haze": round(facts.haze, 2),
                        "tonal": {"range": round(tonal.rng, 2), "median": round(tonal.p50, 2), "sky_minus_foreground": round((tonal.sky_med or 0) - (tonal.fg_med or 0), 2) if tonal.sky_med is not None and tonal.fg_med is not None else None,
                                  "adaptive_strength": round(k_adapt, 2)}}


def _keep_current(p: EditParams, cur: EditParams, touched: List[str], kk: float):
    """Blend the look with the edit that was on the photo: weight kk on the look, 1-kk on what was there."""
    import match
    p.basic.keep_whites = cur.basic.keep_whites                  # a setting, not an adjustment: never summed
    keep = 1.0 - kk
    if keep <= 1e-6:
        return
    for key in type(p.basic).model_fields:
        if key == "keep_whites":
            continue
        if key not in ("temperature", "tint"):
            setattr(p.basic, key, getattr(p.basic, key) + keep * getattr(cur.basic, key))
    for c in p.mixer:
        p.mixer[c] = [p.mixer[c][i] + keep * cur.mixer[c][i] for i in range(3)]
    for key in type(p.curve).model_fields:
        setattr(p.curve, key, getattr(p.curve, key) + keep * getattr(cur.curve, key))
    for n, r in p.regions.items():
        for a in ADJ:
            setattr(r, a, getattr(r, a) + (keep if n in touched else 1.0) * getattr(cur.regions[n], a))
    for name in ("shadows", "mids", "highs", "glob"):
        w, w0 = getattr(p.grading, name), getattr(cur.grading, name)
        v = match._wheel_vec(w) + keep * match._wheel_vec(w0)
        match._set_wheel(w, v, 60.0)
        w.lum = w.lum + keep * w0.lum
    p.grading.balance += keep * cur.grading.balance
    for fld in ("dodge", "burn", "separation", "form"):
        setattr(p.dodge_burn, fld, getattr(p.dodge_burn, fld) * kk + getattr(cur.dodge_burn, fld) * keep)
    p.effects.vignette = p.effects.vignette + keep * cur.effects.vignette


def _govern_exposure(an, p: EditParams, base: Optional[EditParams] = None) -> List[str]:
    """Keep the overall brightness in a believable range and stop highlights clipping / shadows crushing."""
    notes: List[str] = []
    src, _, _ = engine._target(an, 480)
    ref = np.asarray(engine.render(an, base, 480), np.float32) / 255 if base is not None else src     # judge the look against the edit it starts from
    base_clip = float((ref.max(axis=2) >= 0.995).mean())
    base_crush = float((ref.max(axis=2) <= 0.02).mean())
    for _ in range(3):
        out = np.asarray(engine.render(an, p, 480), np.float32) / 255
        med = float(np.median(out @ LUMA))
        clip = float((out.max(axis=2) >= 0.995).mean()) - base_clip
        crush = float((out.max(axis=2) <= 0.02).mean()) - base_crush
        changed = False
        if clip > 0.03:
            p.basic.highlights = max(-60, p.basic.highlights - 8)
            p.basic.whites = min(p.basic.whites, 0) - 3
            changed = True
            notes.append("Highlights were clipping → pulled back")
        if crush > 0.05:
            p.basic.blacks += 6
            p.basic.shadows += 6
            changed = True
            notes.append("Shadows were crushing → opened a little")
        if med < 0.26:
            p.basic.exposure += min(10, (0.30 - med) * 80)
            changed = True
            notes.append("Picture came out dark → brightened")
        elif med > 0.62:
            p.basic.exposure -= min(10, (med - 0.58) * 80)
            changed = True
            notes.append("Picture came out bright → darkened")
        if not changed:
            break
    return notes
