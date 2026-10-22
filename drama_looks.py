"""Dynamic Drama looks: read the photo, then generate the looks that suit it, each with numbers computed from THIS picture.

A fixed preset list treats every photo the same. Here the engine first measures the scene (light, sky, how much of the picture is grey,
which hues are already there, whether there is a bright light source, a horizon, water, depth, people) and then
  1. scores a family of colour moods (blue hour, golden glow, teal & orange, fiery sky, moody silver, pastel dawn, emerald mist)
     for how well each suits this scene and ranks them,
  2. sets every number of each look from the scene: how hard to colour (grey scenes need more), how much sky burn, whether there is
     a light to bloom, how much mist (only with a horizon / depth), how heavy the vignette (lighter with people), how much detail,
  3. builds the whole edit through the same machinery as the Smart looks (regions, lighting, safety governor), keeps the
     as-shot white balance (a pro keeps the mood; neutralising it is what makes skies grey), and checks the result.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import drama
import engine
import look_engine
from look_engine import Feel, Look
from params import Drama, EditParams

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


# ------------------------------------------------------------------------------------------------ what the scene is like
@dataclass
class Scene:
    light: str = "sunny"
    kind: str = "landscape"
    neutral: float = 0.3                 # share of nearly-grey pixels
    warm: float = 0.0                    # share of the coloured pixels that are orange / yellow
    cool: float = 0.0                    # ... blue / teal
    green: float = 0.0                   # ... green
    sky: float = 0.0
    sky_kind: str = "none"
    sky_warm: float = 0.0
    water: float = 0.0
    trees: float = 0.0
    has_horizon: bool = False
    bright_source: float = 0.0           # 0..1: is there a small, very bright light (sun, lamp)?
    haze: float = 0.0
    depth_range: float = 0.0
    people: bool = False
    flat: float = 0.0
    low_key: float = 0.0
    median: float = 0.45
    sky_blue: float = 0.0                # share of the sky that is clearly blue
    sky_luma: float = 0.0                # mean brightness of the sky in the ORIGINAL (a bright flat veil needs modelling)
    leaf_warm: float = 0.0               # share of the TREE pixels that are warm-coloured (autumn leaves), not just warm anywhere
    sun_xy: Tuple[float, float] = (-1.0, -1.0)
    sun: float = 0.0                     # 1 if a sun / lamp is found
    white: float = 0.0                   # brightest level of the ORIGINAL picture (99.5th percentile of luminance)
    chroma: float = 0.0
    bright_grey: float = 0.0             # share of bright near-neutral pixels anywhere (white cloud, snow): colour poured on them turns garish fast
    snow: float = 0.0                    # share of bright, near-neutral pixels (snow, white cloud, pale rock)
    cast_hue: float = 0.0                # the picture's own colour cast (hue of the mids and highlights) and how strong it is
    cast: float = 0.0
    evidence: List[str] = field(default_factory=list)


def scene_preview(an) -> np.ndarray:
    """The picture with a standard tone / contrast / vibrance treatment but its AS-SHOT colour cast (no white-balance neutralising).
    Reading the scene from this, not from the raw pixels, makes a flat RAW and the camera's own punchy JPEG of the same scene read alike."""
    p = engine.auto_params(an, "auto")
    p.wb_mode = "asshot"
    p.basic.temperature = 0.0
    p.basic.tint = 0.0
    for r in p.regions.values():
        for f in ("exposure", "contrast", "highlights", "shadows", "vibrance", "saturation", "clarity", "dehaze", "temperature", "tint", "hue"):
            setattr(r, f, 0.0)
    p.people.enabled = False
    p.dodge_burn.enabled = False
    p.effects.vignette = 0.0
    return np.asarray(engine.render(an, p, 480, True), np.float32) / 255.0


def read_scene(an) -> Scene:
    cached = getattr(an, "_drama_scene", None)
    if cached is not None:
        return cached
    s = _read_scene(an)
    try:
        an._drama_scene = s            # the scene does not change while a photo is open: read it once
    except Exception:
        pass
    return s


def _read_scene(an) -> Scene:
    import sky_engine
    s = Scene(kind=an.scene.kind, light=an.read.light if an.read else "sunny", haze=float(an.scene.haze), people="person" in an.masks)
    a = scene_preview(an)
    y = a @ LUMA
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2Lab)
    C = np.hypot(lab[..., 1], lab[..., 2])
    s.chroma = float(C.mean())
    tone = y > np.percentile(y, 35)
    ca, cb = float(lab[..., 1][tone].mean()), float(lab[..., 2][tone].mean())
    s.cast, s.cast_hue = float(np.hypot(ca, cb)), float(np.degrees(np.arctan2(cb, ca)) % 360)
    s.neutral = float((C < 9).mean())
    s.bright_grey = float(((y > 0.62) & (C < 14)).mean())
    tm = an.masks.get('trees') if getattr(an, 'masks', None) else None
    if tm is not None:
        tmm = cv2.resize(np.asarray(tm, np.float32), (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5
        if tmm.sum() > 300:
            hue_ = (np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360)
            s.leaf_warm = float((((hue_ > 25) & (hue_ < 110) & (C > 10)) & tmm).sum() / tmm.sum())
    sk0 = an.masks.get('sky') if getattr(an, 'masks', None) else None
    notsky = np.ones(y.shape, bool)
    if sk0 is not None:
        notsky = cv2.resize(np.asarray(sk0, np.float32), (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR) < 0.4
    s.snow = float(((y > 0.62) & (C < 14) & notsky).mean())            # bright neutral GROUND (snow); a white sky or a blown sun is not snow
    col = C > 14
    if col.sum() > 100:
        hue = (np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360)[col]
        n = len(hue)
        s.warm = float(((hue > 25) & (hue < 110)).sum() / n)
        s.green = float(((hue >= 110) & (hue < 170)).sum() / n)
        s.cool = float(((hue > 190) & (hue < 300)).sum() / n)
    cov = an.coverage or {}
    s.sky, s.water, s.trees = cov.get("sky", 0) / 100.0, cov.get("water", 0) / 100.0, cov.get("trees", 0) / 100.0
    r = sky_engine.read_sky(an)
    if r is not None:
        s.sky_kind, s.sky_warm = r.kind, r.warm
        s.has_horizon = True
        if r.kind in ("overcast", "dramatic") and r.blue < 0.15 and r.warm < 0.2 and s.light in ("sunny", "general"):
            s.light = "overcast"              # the picture-wide light reader says 'sunny' for any daylight; a grey cloud deck says otherwise
        s.sky_blue = float(r.blue)
    # a small, very bright light source: a sun or a lamp
    hi = y > 0.97
    frac = float(hi.mean())
    s.bright_source = float(np.clip(1.0 - abs(np.log10(max(frac, 1e-5) / 0.01)) / 1.6, 0, 1)) if 0.0003 < frac < 0.12 else 0.0
    if an.depth is not None:
        d = an.depth.astype(np.float32)
        s.depth_range = float(np.percentile(d, 95) - np.percentile(d, 5))
    q = np.percentile(y, [5, 50, 95])
    s.flat = float(np.clip((0.62 - (q[2] - q[0])) / 0.30, 0, 1))
    s.low_key = float(np.clip((0.40 - q[1]) / 0.25, 0, 1))
    s.median = float(q[1])
    src, _, _ = engine._target(an, 480)
    src = np.asarray(src, np.float32)
    sp = drama._find_sun(src)                       # found in the ORIGINAL (the standard preview already has its highlights pulled down)
    sk = an.masks.get('sky') if getattr(an, 'masks', None) else None
    if sk is not None and (sk > 0.5).sum() > 200:
        import cv2 as _cv
        m_ = _cv.resize(np.asarray(sk, np.float32), (src.shape[1], src.shape[0]), interpolation=_cv.INTER_LINEAR) > 0.5
        s.sky_luma = float((src @ LUMA)[m_].mean())
    s.sun = 1.0 if sp is not None else 0.0
    s.sun_xy = (sp[0] / src.shape[1], sp[1] / src.shape[0]) if sp is not None else (-1.0, -1.0)
    s.white = float(np.percentile(np.asarray(src, np.float32)[::2, ::2] @ LUMA, 99.5))
    return s


# ------------------------------------------------------------------------------------------------ the family of looks
@dataclass
class Variant:
    id: str
    label: str
    blurb: str
    hues: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]    # (hue, chroma) for shadows, mids, highlights
    bloom_hue: float
    feel: Feel
    basic: Dict[str, float]
    vignette: float
    mood: float = 85
    protect: float = 60
    enrich: float = 25
    sky_burn: float = 60
    sky_texture: float = 50
    sky_colour: float = 40
    sky_model: float = 0
    bloom: float = 30
    mist: float = 30
    detail: float = 55
    sky_hue: float = 60
    sky_chroma: float = 0
    depth_haze: float = 25
    glow: float = 30
    rays: float = 0
    mixer: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)      # HSL mixer nudges (hue, saturation, luminance) per colour


# global style knobs, tuned against real pro before/after pairs (see tune_drama.py)
TUNE = dict(white=1.0, hl=0.5, vib=0.0, con=0.0, chroma=1.3, enrich=1.3, burn=0.6, mist=0.6, bloom=1.0, expo=1.0, lift=0.0)

VARIANTS: Dict[str, Variant] = {v.id: v for v in [
    Variant("blue_hour", "Blue Hour Drama", "Cold, cohesive blue; dark brooding sky; mist on the horizon",
            ((275, 14), (273, 24), (268, 20)), 255, Feel(-0.4, 0.8, 0.8, 0.0, 0.3),
            dict(contrast=8, highlights=-26, shadows=6, blacks=-4, whites=-4, exposure=-6), -40, sky_burn=75, sky_texture=60, mist=55, bloom=18, enrich=15, sky_hue=272, sky_chroma=26, sky_colour=55, glow=15, rays=0, sky_model=55),
    Variant("golden_glow", "Golden Glow", "Warm amber poured over everything, glowing light, lush greens kept",
            ((60, 12), (72, 28), (78, 34)), 72, Feel(0.9, 0.4, 0.6, 0.6, 0.5),
            dict(contrast=10, highlights=-22, shadows=10, blacks=-8, whites=2, exposure=-3), -30, sky_burn=45, sky_texture=40, mist=28, bloom=70, protect=70, enrich=35, sky_hue=62, sky_chroma=30, sky_colour=55, glow=75, rays=55, sky_model=35),
    Variant("teal_orange", "Teal & Orange", "Cinematic complementary pair: teal shadows, warm highlights",
            ((225, 14), (60, 10), (62, 30)), 65, Feel(0.3, 0.5, 0.7, 0.3, 0.4),
            dict(contrast=14, highlights=-18, shadows=8, blacks=-10, whites=0, exposure=-3), -30, mood=80, protect=65, sky_burn=50, bloom=35, mist=25, enrich=30, glow=45, rays=28, sky_model=45),
    Variant("fiery_sky", "Fiery Sky", "Orange and magenta sky with a violet-cool foreground",
            ((300, 16), (25, 26), (55, 34)), 50, Feel(0.8, 0.8, 0.7, 0.5, 0.3),
            dict(contrast=14, highlights=-30, shadows=8, blacks=-10, whites=-2, exposure=-6), -34, sky_burn=60, sky_texture=60, sky_colour=85, bloom=60, mist=30, enrich=30, sky_hue=30, sky_chroma=34, glow=60, rays=38, sky_model=45),
    Variant("moody_silver", "Moody Silver", "Dark, desaturated cool grey; heavy vignette; quiet drama",
            ((268, 8), (268, 8), (262, 6)), 250, Feel(-0.2, 1.0, 0.8, -0.5, 0.0),
            dict(contrast=18, highlights=-24, shadows=0, blacks=-10, whites=-6, exposure=-12), -46, mood=75, protect=40, sky_burn=70, sky_texture=60, mist=40, bloom=10, enrich=-10, glow=0, rays=0, sky_model=50),
    Variant("pastel_dawn", "Pastel Dawn", "Soft pink-lilac light over teal shadows, lifted and airy",
            ((215, 10), (335, 10), (10, 14)), 5, Feel(0.3, 0.1, 0.2, -0.2, 0.6),
            dict(contrast=2, highlights=-14, shadows=12, blacks=6, whites=-4, exposure=0), -18, mood=70, protect=55, sky_burn=30, sky_texture=35, sky_colour=55, bloom=45, mist=45, enrich=10, detail=35, glow=40, rays=15, sky_model=20),
    Variant("emerald_mist", "Emerald Mist", "Deep green-gold forest light with a hint of fog",
            ((190, 12), (125, 18), (92, 26)), 90, Feel(0.4, 0.5, 0.7, 0.5, 0.4),
            dict(contrast=10, highlights=-20, shadows=10, blacks=-6, whites=0, exposure=-4), -32, mood=75, protect=75, sky_burn=45, bloom=40, mist=60, enrich=35, glow=35, rays=22, sky_model=30),
    Variant("alpine_blue", "Alpine Blue", "Deep blue sky and shadows with clouds that keep their drama; warm glowing foreground",
            ((290, 14), (282, 20), (275, 8)), 255, Feel(-0.1, 0.7, 0.9, 0.4, 0.5),
            dict(contrast=10, highlights=-24, shadows=10, blacks=-8, whites=-2, exposure=-6), -28, mood=75, protect=65, enrich=45, sky_burn=45, sky_texture=80, bloom=25, mist=15,
            sky_hue=278, sky_chroma=28, sky_colour=70, glow=20, rays=10, sky_model=70),
    Variant("autumn_gold", "Autumn Gold", "Amber and copper foliage glowing against cool, shaded shadows; greens turned olive-gold, a hint of forest mist",
            ((215, 10), (62, 24), (72, 32)), 70, Feel(0.8, 0.5, 0.7, 0.5, 0.4),
            dict(contrast=10, highlights=-18, shadows=10, blacks=-8, whites=0, exposure=-3), -28, mood=72, protect=72, enrich=45, sky_burn=45, sky_texture=45,
            sky_colour=30, bloom=40, mist=40, detail=62, sky_hue=60, sky_chroma=0, glow=40, rays=25, sky_model=30,
            mixer={"orange": (0.0, 14.0, 4.0), "yellow": (-6.0, 12.0, 3.0), "red": (0.0, 10.0, 0.0), "green": (-14.0, -6.0, -2.0)}),
    Variant("lush_valley", "Lush Valley", "Vivid layered greens under a warm orange sky, haze deepening the distance",
            ((130, 16), (118, 30), (92, 30)), 70, Feel(0.6, 0.4, 0.9, 0.9, 0.3),
            dict(contrast=8, highlights=-20, shadows=10, blacks=-8, whites=-2, exposure=-6), -22, mood=55, protect=90, enrich=60, sky_burn=35, sky_texture=45, bloom=45, mist=45, detail=75,
            sky_hue=58, sky_chroma=36, sky_colour=85, glow=45, rays=28, sky_model=30),
]}



def score(v: Variant, s: Scene) -> Tuple[float, str]:
    """How well a look suits this scene (0..1), with the reason in plain words."""
    sc, why = 0.35, ""

    def add(x, reason):
        nonlocal sc, why
        sc += x
        if x >= 0.12 and not why:
            why = reason
    lt = s.light
    fol = s.green * min(1.0, s.chroma / 18.0) * float(np.clip((s.median - 0.18) / 0.20, 0.35, 1.0))                 # real green colour, not just 'trees' (bare winter trees are not foliage)
    sn = min(1.0, s.snow * 4.0)                              # snow scene: keep it white / blue, never yellow or green
    if sn > 0.3 and v.id in ('golden_glow', 'emerald_mist', 'lush_valley', 'fiery_sky'):
        sc -= 0.6 * sn
    if sn > 0.3 and v.id in ('blue_hour', 'moody_silver', 'alpine_blue'):
        add(0.2 * sn, 'snow and cold light that suit a cool, quiet grade')
    if s.cast > 6:                       # a strong colour cast is the strongest hint: pros amplify the mood that is already there
        mid_h = v.hues[1][0]
        diff = abs((mid_h - s.cast_hue + 180) % 360 - 180)
        if diff < 45:
            add(0.12 + 0.3 * min(1.0, s.cast / 14.0) * (1 - diff / 45), "it already has that colour cast, which this look deepens")
        elif diff > 110:
            sc -= 0.2
    if v.id == "blue_hour":
        if lt in ("blue_hour", "night"): add(0.45, "blue-hour light")
        if lt == "overcast" or s.sky_kind in ("overcast", "dramatic", "hazy"): add(0.25, "a dull, cool sky that suits deep blue")
        if s.water > 0.08: add(0.12, "water that takes a cold mood well")
        if s.cool > 0.4: add(0.15, "colours that are already cool")
        if s.warm > 0.5 and lt in ("golden_hour", "sunset"): sc -= 0.25
    elif v.id == "golden_glow":
        if lt in ("golden_hour", "sunset"): add(0.4, "low, warm light")
        if s.sky_warm > 0.2: add(0.15, "warm tones already in the sky")
        if s.bright_source > 0.4: add(0.2, "a bright light to bloom")
        if fol > 0.2: add(0.1, "foliage that glows in gold")
        if s.cool > 0.6 and lt not in ("golden_hour", "sunset"): sc -= 0.1
    elif v.id == "teal_orange":
        add(0.18, "a safe, cinematic all-rounder")
        if lt == 'overcast' and s.sky_blue < 0.15: sc -= 0.2                # warm-tinting a grey cloud deck turns it pink
        if s.warm > 0.25 and s.cool > 0.15: add(0.2, "warm and cool already side by side")
        if s.sky > 0.15 and s.sky_kind in ("clear", "scattered"): add(0.12, "blue sky against warm land")
    elif v.id == "fiery_sky":
        if s.sky_kind == "sunset": add(0.5, "a sunset sky")
        elif s.sky > 0.25 and s.sky_kind in ("dramatic", "scattered", "clear"): add(0.18, "a large sky with room for fire")
        if lt in ("sunset", "golden_hour"): add(0.15, "sunset light")
        if s.sky < 0.1: sc -= 0.2
    elif v.id == "moody_silver":
        if lt == "overcast" or s.sky_kind == "overcast": add(0.35, "flat grey light that suits a quiet, dark mood")
        if s.sky_luma > 0.8 and s.sky > 0.3: sc -= 0.3                   # a bright cloud sky should gain depth and colour, not be turned grey
        if s.neutral > 0.55: add(0.15, "a mostly neutral scene")
        if s.low_key > 0.4: add(0.1, "already dark")
        if s.chroma > 28: sc -= 0.15
    elif v.id == "pastel_dawn":
        if s.sky_kind in ("clear", "hazy") or s.haze > 0.3: add(0.2, "soft, hazy light")
        if s.water > 0.12: add(0.15, "calm water to carry soft colour")
        if s.low_key > 0.5: sc -= 0.15
        if s.flat > 0.4: add(0.12, "flat light that benefits from a soft grade")
    elif v.id == "emerald_mist":
        if fol > 0.35: add(0.22, "a lot of foliage")
        if s.haze > 0.25 or s.depth_range > 0.5: add(0.12, "depth that fog will deepen")
        if fol < 0.1: sc -= 0.25
    elif v.id == "alpine_blue":
        if s.sky > 0.25 and s.sky_kind in ("overcast", "dramatic", "scattered"): add(0.18, "a big, cloudy sky that deserves contrast")
        if s.neutral > 0.6 and s.sky > 0.3: add(0.15, "a grey, bright sky to bring colour to")
        if s.sky_luma > 0.8 and s.sky > 0.3: add(0.3, "a bright cloud sky that deserves depth and blue")
        if s.bright_source > 0.5: add(0.05, "light breaking through")
    elif v.id == "lush_valley":
        if fol > 0.3 and s.warm > 0.3: add(0.3, "green land under warm light")
        if s.depth_range > 0.45 or s.haze > 0.2: add(0.18, "layers of distance and haze")
        if fol < 0.1: sc -= 0.35                                      # 'lush' needs real green
        if lt in ("sunset", "golden_hour"): add(0.15, "warm low light")
    autumn = s.trees > 0.15 and s.leaf_warm > 0.28 and s.warm > 0.2 and s.snow < 0.15
    if v.id == 'autumn_gold' and autumn: add(0.5 + 0.2 * min(1.0, s.warm), 'foliage turning gold: amber and copper glow')
    if v.id == 'autumn_gold' and not autumn: sc -= 0.25
    if s.trees > 0.3 and s.warm > 0.22 and s.green < 0.65:                # autumn-like foliage: keep the warmth, do not cool it
        if v.id in ('golden_glow', 'lush_valley', 'teal_orange'): add(0.14, 'foliage turning gold')
        if v.id in ('blue_hour', 'moody_silver'): sc -= 0.12
    if s.sun > 0.5 and v.id in ('golden_glow', 'fiery_sky', 'teal_orange', 'lush_valley'):
        add({'golden_glow': 0.5, 'fiery_sky': 0.12, 'teal_orange': 0.05, 'lush_valley': 0.08}[v.id], 'a sun to build the light around')
    if v.id == "golden_glow" and s.bright_source > 0.4 and s.low_key > 0.3: add(0.22, "a bright sun in a dark scene")
    if s.people:
        sc -= 0.04
        if v.id in ('blue_hour', 'moody_silver'): sc -= 0.15           # cold grades make people look unwell and a sunny day drab
        if v.id == 'alpine_blue': sc -= 0.22                                    # a heavy blue sky grade is a landscape look: it greys and purples the people in front of it
        if v.id in ('fiery_sky', 'emerald_mist', 'lush_valley'): sc -= 0.10      # strong sky / foliage colour fights skin tones
        if v.id == 'golden_glow' and lt in ('overcast', 'sunny', 'general') and s.sky_blue < 0.2:
            add(0.36, 'warm light that flatters skin under a flat, grey sky')              # warmth on skin is what lifts a portrait under a dull sky
        if v.id == 'pastel_dawn' and lt == 'overcast': add(0.18, 'soft, lifted light that suits people under cloud')
    return float(np.clip(sc, 0.05, 1.0)), why or "a good starting point"


def generate(an) -> List[dict]:
    """The Drama looks for this photo, ranked. Each: id, label, blurb, fit, reason."""
    s = read_scene(an)
    out = []
    for v in VARIANTS.values():
        f, why = score(v, s)
        out.append({"id": v.id, "label": v.label, "blurb": v.blurb, "fit": round(f, 2), "reason": why})
    out.sort(key=lambda d: -d["fit"])
    for i, d in enumerate(out):
        d["recommended"] = i < 3
    return out


# ------------------------------------------------------------------------------------------------ building one look for THIS photo
def _adapt(v: Variant, s: Scene) -> Dict[str, float]:
    """The numbers of a look, set from the scene."""
    grey = float(np.clip(0.55 + 0.6 * s.neutral + 0.25 * (1 - min(s.chroma / 30.0, 1)), 0.5, 1.15))
    out = dict(
        mood=float(np.clip(v.mood * grey, 35, 100)),
        sky_burn=v.sky_burn * (1.0 if s.has_horizon and s.sky > 0.06 else 0.0) * float(np.clip(0.7 + s.sky * 1.5, 0.7, 1.1)),
        sky_texture=v.sky_texture * (1.0 + 0.4 * s.flat),
        sky_colour=v.sky_colour * (0.7 if s.sky_kind in ("sunset",) else 1.0) * (1.0 - 0.6 * float(np.clip((s.sky_luma - 0.7) / 0.25, 0, 1))),   # white clouds stay white: colour them lightly
        sky_model=v.sky_model * float(np.clip((s.sky_luma - 0.55) / 0.3, 0, 1)) * (1.0 if s.has_horizon and s.sky > 0.08 else 0.0),
        bloom=v.bloom * (0.3 + 0.7 * s.bright_source),
        mist=v.mist * (0.55 + 0.45 * float(np.clip(s.water * 3 + s.sky * 1.5, 0, 1))) * (1.0 if s.has_horizon else 0.0),
        depth_haze=v.depth_haze * 0.6 * float(np.clip(1.1 - 2.8 * s.haze, 0.1, 1.0)) * (1.0 if s.depth_range > 0.3 else 0.5),
        vignette=v.vignette * (0.55 if s.people else 1.0),
        detail=v.detail * (1.0 + 0.5 * s.flat),
        enrich=v.enrich,
        glow=v.glow * max(float(np.clip((s.bright_source - 0.2) / 0.5, 0, 1)), 0.8 * s.sun),
        rays=v.rays * max(float(np.clip((s.bright_source - 0.3) / 0.4, 0, 1)), 0.8 * s.sun),
    )
    if s.people:
        out["mood"] *= 0.55                                   # a person's skin and clothes must stay believable: grade the scene, not the subject
        out["enrich"] = min(out["enrich"], 8)
        out["sky_colour"] *= 0.7
    return out


def build(an, base: EditParams, vid: str, strength: float = 1.0) -> Tuple[EditParams, List[str], dict]:
    """The photo with Drama look `vid` applied at `strength` (0..1.5). Returns (params, notes, facts)."""
    v = VARIANTS[vid]
    s = read_scene(an)
    a = _adapt(v, s)
    notes: List[str] = []
    # Drama owns the sky darkening (gradient, texture kept), so the region step must not darken the sky a second time; and a scene that is
    # already bright must not be pushed darker (a pro lowers a blown sky by ~20%, no more)
    feel = Feel(v.feel.warm, v.feel.depth * 0.4, v.feel.punch, v.feel.colour, v.feel.lift)
    basic = dict(v.basic)
    for k_ in ('highlights', 'whites'):
        if basic.get(k_, 0) < 0:
            basic[k_] = basic[k_] * TUNE['hl']
    if basic.get('exposure', 0) < 0:
        basic['exposure'] = basic['exposure'] * TUNE['expo']
    if s.median > 0.5:
        basic["exposure"] = min(0.0, basic.get("exposure", 0.0) * max(0.0, 1.0 - (s.median - 0.5) * 6))
        basic["highlights"] = basic.get("highlights", 0.0) * 0.6
    look = Look(id=f"drama_{v.id}", label=v.label, result=v.blurb, adjusts=v.blurb, feel=feel, basic=basic, wheels={}, mixer=dict(v.mixer), curve={}, vignette=a["vignette"])
    p, lnotes, facts = look_engine.build(an, base, look, strength)
    p.basic.highlights = max(p.basic.highlights, -26.0)      # a gentler, more basic starting point: Drama and the white anchor do the light, the sliders do not crush it
    p.basic.shadows = min(p.basic.shadows, 26.0)
    # the look starts from a clean tone slate, which throws away what Auto MEASURED about this photo (a dark scene needs lifting before it is graded):
    # carry the measured exposure and shadow corrections over, so a dark photo is not made darker and a bright one is not blown
    p.basic.exposure += 0.85 * base.basic.exposure
    p.basic.shadows = max(p.basic.shadows, 0.85 * base.basic.shadows)
    if s.median < 0.34:
        p.basic.exposure += (0.34 - s.median) * 55.0 * min(float(np.clip(strength, 0.0, 1.5)), 1.0)               # a low-key scene (not on purpose: no bright light) gets lifted toward a readable median
        p.basic.blacks = max(p.basic.blacks, -4.0)
    p.wb_mode = "asshot"                                   # keep the capture's mood: neutralising it is what makes skies grey
    p.basic.temperature = 0.0
    p.basic.tint = 0.0
    k = float(np.clip(strength, 0.0, 1.5))
    d = p.drama
    d.enabled = True
    d.name = v.label
    d.amount = 100 * k if k > 0 else 0
    d.mood, d.protect, d.enrich = a["mood"], v.protect, a["enrich"] * TUNE['enrich']
    (d.sh_hue, d.sh_chroma), (d.mid_hue, d.mid_chroma), (d.hi_hue, d.hi_chroma) = v.hues
    sn = min(1.0, s.bright_grey * 3.0)
    cs = (1.0 + 0.2 * s.neutral) * (1.0 - 0.45 * sn) * TUNE['chroma']                                  # a grey scene needs a little more colour to feel graded
    d.sh_chroma, d.mid_chroma, d.hi_chroma = d.sh_chroma * cs, d.mid_chroma * cs, d.hi_chroma * cs * (1.0 - 0.4 * sn)
    d.sky_hue, d.sky_chroma = v.sky_hue, v.sky_chroma * (0.7 if s.people else 1.0)
    if s.people:
        d.sh_chroma, d.mid_chroma, d.hi_chroma = d.sh_chroma * 0.7, d.mid_chroma * 0.7, d.hi_chroma * 0.7
    if 195 <= d.sh_hue <= 310 and s.neutral > 0.45:                        # grey scene + cool look: the dark areas take deep, saturated blue (pro edits do)
        d.sh_chroma = min(46.0, d.sh_chroma * (1.0 + 1.6 * min(1.0, (s.neutral - 0.45) / 0.4)))
    d.sky_model = a['sky_model']
    d.sky_burn, d.sky_texture, d.sky_colour = a["sky_burn"] * TUNE['burn'] * (1.0 - 0.8 * s.bright_source), a["sky_texture"], a["sky_colour"]
    d.bloom, d.bloom_hue, d.mist, d.mist_y, d.detail = a["bloom"] * TUNE['bloom'], v.bloom_hue, a["mist"] * TUNE['mist'], -1.0, a["detail"]
    d.depth_haze = a["depth_haze"]
    d.glow, d.rays = a['glow'], a['rays']
    d.sun_x, d.sun_y = s.sun_xy
    d.white = float(np.clip(s.white, 0.0, 0.97)) * TUNE['white'] if s.white > 0.6 else 0.0
    # step 4: stronger local contrast and dehaze, through the ordinary (halo-guarded) Basic controls
    dk = a["detail"] / 55.0 * min(k, 1.2)
    p.basic.dehaze += (16 + 40 * s.haze) * dk
    p.basic.clarity += 14 * dk
    p.basic.texture += 10 * dk
    p.basic.vibrance += TUNE['vib'] + 16.0 * float(np.clip((s.neutral - 0.3) / 0.4, 0, 1))      # a grey scene needs a real colour push (pro edits carry 2-4x the colour)
    p.basic.contrast += TUNE['con']
    p.basic.shadows += TUNE['lift']
    notes.append(f"{v.label}: {v.blurb}")
    notes.append("As-shot white balance kept (the mood is the point)")
    if s.neutral > 0.5:
        notes.append(f"{s.neutral * 100:.0f}% of the picture is nearly grey → colour poured in more strongly")
    if a["bloom"] < 0.5 * v.bloom:
        notes.append("No small bright light found → little bloom")
    elif a["bloom"] > 0.8 * v.bloom and v.bloom > 30:
        notes.append("A bright light source found → bloom around it")
    if not s.has_horizon:
        notes.append("No sky found → sky burn skipped")
    if s.people:
        notes.append("People in the frame → colour softened, vignette lighter, skin protected")
    notes += [n for n in lnotes if n.startswith(("Region", "Haze", "Snow", "Light already"))][:2]
    notes += _govern(an, p, base)
    return p, notes, {"drama": v.id, "hues": [list(h) for h in v.hues], "regions": facts.get("regions", [])}


def _govern(an, p: EditParams, base: EditParams) -> List[str]:
    """Test-render small; if the look went too far (clipped highlights, crushed shadows, skin shifted, far too dark), ease it back."""
    notes: List[str] = []
    src, size, _ = engine._target(an, 480)
    skin_m = engine._masks_at(an, size).get("skin")
    skin = skin_m > 0.6 if skin_m is not None and (skin_m > 0.6).sum() > 200 else None
    ref = np.asarray(engine.render(an, base, 480), np.float32) / 255
    clip0, crush0 = float((ref.max(2) > 0.995).mean()), float((ref.max(2) < 0.02).mean())
    for _ in range(3):
        out = np.asarray(engine.render(an, p, 480), np.float32) / 255
        clip, crush = float((out.max(2) > 0.995).mean()) - clip0, float((out.max(2) < 0.02).mean()) - crush0
        med = float(np.median(out @ LUMA))
        skin_shift = float(np.abs(out[skin] - ref[skin]).mean()) if skin is not None else 0.0
        c_out = float(np.hypot(*np.moveaxis(cv2.cvtColor(out, cv2.COLOR_RGB2Lab)[..., 1:], -1, 0)).mean())
        c_ref = float(np.hypot(*np.moveaxis(cv2.cvtColor(ref, cv2.COLOR_RGB2Lab)[..., 1:], -1, 0)).mean())
        bad = []
        if clip > 0.05: bad.append("highlights were clipping")
        if crush > 0.10: bad.append("shadows were crushing")
        if med < 0.14: bad.append("it came out very dark")
        if skin_shift > 0.12: bad.append("skin shifted too much")
        if c_out > c_ref * 2.4 + 9: bad.append("colour was getting garish")
        if not bad:
            break
        p.drama.amount *= 0.8
        p.drama.mood *= 0.85
        p.drama.bloom *= 0.7
        if "colour was getting garish" in bad:
            p.drama.enrich *= 0.7
            p.drama.sky_colour *= 0.8
            p.drama.hi_chroma *= 0.8
            p.drama.mid_chroma *= 0.85
        p.basic.dehaze *= 0.8
        if "it came out very dark" in bad:
            p.basic.exposure += 6
            p.drama.sky_burn *= 0.8
        notes.append("Eased back: " + ", ".join(bad))
    return notes
