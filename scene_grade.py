"""Scene- and light-aware colour grading.

Step 1  read the scene granularly:  LIGHT (golden hour, sunset, blue hour, midday sun, overcast, night, indoor)
        and SCENE (mountain lake, mountains, coast, forest, countryside, urban, snow, desert, portrait ...).
        Light comes mainly from the pixels (warmth of the highlights, the sky's hue and saturation, contrast,
        brightness); the capture time and, when the file has GPS, the sun's real elevation act as extra evidence.
        Scene comes from what the segmentation model found (how much sky, water, mountains, trees ...).
Step 2  build a GRADE PLAN from a recipe library: colour-grading wheels, HSL moves, tone, per-region moods,
        the way a colourist grades a golden-hour mountain lake differently from an overcast forest.
Step 3  the engine writes the plan into ordinary slider values, so every part stays editable, and the user's
        "Colour grading" strength simply scales it.

Every recipe is deliberately restrained (warm highlights, cool shadows, deeper water, controlled greens): the goal is
a graded look, not a filter. The user can override the detected light and scene.
"""
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
LIGHTS = ["golden_hour", "sunset", "blue_hour", "sunny", "overcast", "night", "indoor"]
SCENES = ["mountain_lake", "mountains", "coast", "forest", "countryside", "urban", "snow", "desert", "general"]
LIGHT_LABEL = {"golden_hour": "Golden hour", "sunset": "Sunset glow", "blue_hour": "Blue hour", "sunny": "Midday sun",
               "overcast": "Overcast", "night": "Night", "indoor": "Indoor"}
SCENE_LABEL = {"mountain_lake": "Mountain lake", "mountains": "Mountains", "coast": "Coast / water", "forest": "Forest",
               "countryside": "Countryside", "urban": "Urban", "snow": "Snow", "desert": "Desert / sand", "general": "General"}


# ------------------------------------------------------------------------------------------------ analysis
@dataclass
class SceneRead:
    light: str = "sunny"
    light_conf: float = 0.0
    light_scores: Dict[str, float] = field(default_factory=dict)
    scene: str = "general"
    scene_conf: float = 0.0
    evidence: List[str] = field(default_factory=list)
    sun_elevation: Optional[float] = None

    def summary(self) -> dict:
        return {"light": self.light, "light_label": LIGHT_LABEL[self.light], "light_conf": round(self.light_conf, 2),
                "scene": self.scene, "scene_label": SCENE_LABEL[self.scene], "scene_conf": round(self.scene_conf, 2),
                "evidence": self.evidence, "lights": LIGHTS, "scenes": SCENES,
                "sun_elevation": None if self.sun_elevation is None else round(self.sun_elevation, 1)}


def _hours(when: Optional[dt.datetime]) -> Optional[float]:
    return None if when is None else when.hour + when.minute / 60.0


def read_light(work: np.ndarray, sky: Optional[np.ndarray], indoor: bool, meta) -> Tuple[str, float, Dict[str, float], List[str]]:
    """Classify the light from the as-shot picture (warmth is the signal, so no white balance is applied)."""
    s = min(1.0, 500 / max(work.shape[:2]))
    a = cv2.resize(work, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else work
    y = a @ LUMA
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    skym = None
    if sky is not None:
        skym = cv2.resize(sky.astype(np.float32), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA) > 0.6
        if skym.sum() < 0.01 * skym.size:
            skym = None
    ev: List[str] = []
    nosky = ~skym if skym is not None else np.ones(y.shape, bool)

    # Illuminant colour is read from NEAR-NEUTRAL pixels only (rock, soil, cloud, concrete). Saturated subject matter
    # (yellow flowers, sunlit grass, a red jacket) is warm whatever the light is doing.
    neutral = nosky & (hsv[..., 1] < 0.30) & (y > 0.12)
    pool = neutral if neutral.sum() > 0.01 * y.size else nosky
    hi = pool & (y >= np.percentile(y[pool], 80)) if pool.any() else np.ones(y.shape, bool)
    warm_hi = float(np.log2((a[..., 0][hi].mean() + 1e-3) / (a[..., 2][hi].mean() + 1e-3))) if neutral.sum() > 0.01 * y.size else 0.0
    luma, p5, p95 = float(y.mean()), float(np.percentile(y, 5)), float(np.percentile(y, 95))
    contrast = p95 - p5
    sky_sat = sky_luma = sky_warm = sky_blue = 0.0
    bot_sat = bot_warm = 0.0
    if skym is not None:
        sky_sat = float(hsv[..., 1][skym].mean()); sky_luma = float(y[skym].mean())
        hh = hsv[..., 0][skym]
        sky_blue = float(((hh > 190) & (hh < 260) & (hsv[..., 1][skym] > 0.12)).mean())
        # the horizon end of the sky (lowest 25% of the sky's rows) is where sunset colour lives
        rows = np.nonzero(skym.any(axis=1))[0]
        cut = rows.min() + int(0.75 * (rows.max() - rows.min() + 1))
        bot = skym.copy(); bot[:cut] = False
        if bot.sum() > 30:
            hb, sb = hsv[..., 0][bot], hsv[..., 1][bot]
            bot_sat = float(sb.mean())
            bot_warm = float((((hb < 55) | (hb > 320)) & (sb > 0.18)).mean())
        sky_warm = bot_warm
    dark = y < np.percentile(y, 20)
    shadow_blue = float(np.log2((a[..., 2][dark].mean() + 1e-3) / (a[..., 0][dark].mean() + 1e-3)))

    sc = {k: 0.0 for k in LIGHTS}
    if indoor:
        sc["indoor"] = 0.9
    sc["night"] = float(np.clip((0.16 - luma) / 0.10, 0, 1)) * float(p95 < 0.65)
    sc["golden_hour"] = float(np.clip((warm_hi - 0.12) / 0.30, 0, 1)) * 0.75 + 0.25 * float(np.clip(sky_warm * 2, 0, 1))
    sc["sunset"] = float(np.clip((bot_warm - 0.22) / 0.35, 0, 1)) * float(np.clip((bot_sat - 0.10) / 0.18, 0, 1)) * 1.15
    sc["blue_hour"] = float(np.clip((shadow_blue - 0.15) / 0.3, 0, 1)) * float(np.clip((0.40 - luma) / 0.2, 0, 1)) * float(warm_hi < 0.12)
    grey_sky = skym is not None and sky_sat < 0.17 and sky_blue < 0.5
    sc["overcast"] = (0.55 if grey_sky else 0.0) * float(contrast < 0.70) + 0.35 * float(np.clip((0.50 - contrast) / 0.2, 0, 1))
    sc["overcast"] *= float(1 - np.clip((bot_warm - 0.15) * 2.0, 0, 0.9))      # a warm horizon is not an overcast sky
    sc["sunny"] = 0.15 + 0.50 * float(np.clip((sky_sat - 0.15) / 0.2, 0, 1)) + 0.30 * float(np.clip((contrast - 0.60) / 0.25, 0, 1))
    sc["sunny"] *= float(np.clip(1.3 - 2.0 * max(warm_hi - 0.30, 0), 0.2, 1))        # strongly warm light is not "midday"
    overcast_sky = grey_sky

    # time of day and (if GPS) the sun's real elevation: extra evidence, never decisive on its own
    elev = getattr(meta, "sun_elevation", None)
    hr = _hours(getattr(meta, "when", None))
    if elev is not None:
        ev.append(f"sun {elev:+.0f}° above the horizon ({meta.sun_source})")
        if 0 <= elev < 12:
            sc["golden_hour"] += 0.45; sc["sunset"] += 0.15
        elif -6 <= elev < 0:
            sc["sunset"] += 0.35; sc["blue_hour"] += 0.3
        elif elev < -6:
            sc["blue_hour"] += 0.25; sc["night"] += 0.35
        elif elev > 35:
            sc["sunny"] += 0.30
    elif hr is not None:
        if hr >= 18.5 or hr < 6.5:
            sc["golden_hour"] += 0.10; sc["sunset"] += 0.05
        elif 10.5 <= hr <= 15.5:
            sc["sunny"] += 0.08
        ev.append(f"taken at {int(hr):02d}:{int((hr % 1) * 60):02d}")
    if warm_hi > 0.2:
        ev.append(f"warm light on neutral surfaces (R/B ×{2 ** warm_hi:.2f})")
    if sky_warm > 0.3:
        ev.append("orange/pink glow at the horizon")
    if overcast_sky:
        ev.append("flat grey sky")
    best = max(sc, key=sc.get)
    top2 = sorted(sc.values(), reverse=True)[:2]
    conf = float(np.clip(0.4 + (top2[0] - top2[1]) * 1.4, 0.3, 1.0))
    return best, conf, sc, ev


def read_scene(shares: Dict[str, float], work: np.ndarray, indoor: bool, kind: str, masks: Optional[Dict[str, np.ndarray]] = None) -> Tuple[str, float, List[str]]:
    """Classify the place from what the segmentation model found (shares are percent of the frame)."""
    g = lambda k: shares.get(k, 0.0) / 100.0
    sky, trees, grass, water, mount, build, ground = (g(k) for k in ("sky", "trees", "grass", "water", "mountains", "buildings", "ground"))
    ev: List[str] = []
    s = min(1.0, 400 / max(work.shape[:2]))
    a = cv2.resize(work, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else work
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    bright = (a @ LUMA > 0.72) & (hsv[..., 1] < 0.14)
    if masks is not None and "mountains" in masks:
        # snow lies on mountains / ground: a pale grey sky or a white shirt is not snow
        def at(k):
            m = masks.get(k)
            return None if m is None else cv2.resize(np.asarray(m, np.float32), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        land = at("mountains")
        if at("ground") is not None:
            land = np.maximum(land, at("ground"))
        keep = land > 0.5
        for k in ("sky", "person"):
            if at(k) is not None:
                keep &= at(k) < 0.3
        snowish = float((bright & keep).sum()) / max(1.0, float((land > 0.5).sum())) * float(keep.mean() * 4 if keep.mean() < 0.25 else 1.0)
        snowish = float(min(snowish, 1.0)) * (1.0 if keep.mean() > 0.04 else 0.0)
    else:
        snowish = float(bright[a.shape[0] // 4:].mean())
    sc = {
        "mountain_lake": 2.2 * min(mount, 0.35) + 1.8 * min(water, 0.4) if (mount > 0.06 and water > 0.08) else 0.0,
        "mountains": 2.4 * mount if mount > 0.10 else 0.0,
        "coast": 2.2 * water + 0.8 * ground if (water > 0.18 and mount < 0.08) else 0.0,
        "forest": 2.4 * trees if trees > 0.28 else 0.0,
        "countryside": 2.0 * grass + 0.3 * trees if grass > 0.22 else 0.0,
        "urban": 2.4 * build if build > 0.22 else 0.0,
        "snow": 3.0 * snowish if (snowish > 0.18 and mount > 0.04) else 0.0,
        "desert": 2.2 * ground if (ground > 0.25 and grass < 0.10 and trees < 0.10) else 0.0,
    }
    best = max(sc, key=sc.get)
    if sc[best] < 0.35:
        best = "general"
    conf = float(np.clip(sc.get(best, 0.0) / 1.2, 0.3, 1.0)) if best != "general" else 0.4
    parts = [f"{k} {v * 100:.0f}%" for k, v in sorted(((k, g(k)) for k in ("sky", "mountains", "water", "trees", "grass", "buildings", "ground")),
                                                      key=lambda kv: -kv[1])[:3] if v > 0.03]
    if parts:
        ev.append("mostly " + ", ".join(parts))
    return best, conf, ev


def read(work: np.ndarray, sky: Optional[np.ndarray], shares: Dict[str, float], indoor: bool, kind: str, meta, masks: Optional[Dict[str, np.ndarray]] = None) -> SceneRead:
    light, lc, scores, ev = read_light(work, sky, indoor, meta)
    scene, scn, ev2 = read_scene(shares, work, indoor, kind, masks)
    return SceneRead(light=light, light_conf=lc, light_scores=scores, scene=scene, scene_conf=scn, evidence=ev + ev2,
                     sun_elevation=getattr(meta, "sun_elevation", None))


# ------------------------------------------------------------------------------------------------ recipes
@dataclass
class Plan:
    basic: Dict[str, float] = field(default_factory=dict)                       # added to the basic sliders
    wheels: Dict[str, Tuple[float, float, float]] = field(default_factory=dict) # shadows/mids/highs/glob: hue, sat, lum
    blending: Optional[float] = None
    balance: Optional[float] = None
    mixer: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)  # colour: hue, sat, lum
    regions: Dict[str, Dict[str, float]] = field(default_factory=dict)          # region -> slider deltas
    curve: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _merge(plan: Plan, add: Plan, weight: float = 1.0):
    for k, v in add.basic.items():
        plan.basic[k] = plan.basic.get(k, 0.0) + v * weight
    for k, (h, s_, l) in add.wheels.items():
        if k in plan.wheels and plan.wheels[k][1] > 0 and s_ > 0:
            # blend hues on the circle, weighted by saturation, so two recipes do not fight
            h0, s0, l0 = plan.wheels[k]
            x = s0 * np.cos(np.radians(h0)) + s_ * weight * np.cos(np.radians(h))
            yv = s0 * np.sin(np.radians(h0)) + s_ * weight * np.sin(np.radians(h))
            plan.wheels[k] = (float(np.degrees(np.arctan2(yv, x)) % 360), float(min(100, np.hypot(x, yv))), l0 + l * weight)
        else:
            plan.wheels[k] = (h, s_ * weight, l * weight)
    if add.blending is not None:
        plan.blending = add.blending
    if add.balance is not None:
        plan.balance = (plan.balance or 0) + add.balance * weight
    for k, (h, s_, l) in add.mixer.items():
        h0, s0, l0 = plan.mixer.get(k, (0, 0, 0))
        plan.mixer[k] = (h0 + h * weight, s0 + s_ * weight, l0 + l * weight)
    for r, d in add.regions.items():
        cur = plan.regions.setdefault(r, {})
        for k, v in d.items():
            cur[k] = cur.get(k, 0.0) + v * weight
    for k, v in add.curve.items():
        plan.curve[k] = plan.curve.get(k, 0.0) + v * weight
    plan.notes += add.notes


LIGHT_RECIPES: Dict[str, Plan] = {
    # warm sun on rock and water: orange highlights, teal-blue shadows, rich deep water, olive greens
    "golden_hour": Plan(
        basic=dict(contrast=14, highlights=-20, shadows=12, whites=8, blacks=-14, dehaze=8, clarity=10, vibrance=14),
        wheels=dict(highs=(36, 36, 4), mids=(30, 10, 0), shadows=(208, 28, -2)), blending=60, balance=8,
        mixer=dict(orange=(0, 14, 8), yellow=(-6, -10, -4), green=(-12, -6, -12), aqua=(0, 22, -10), blue=(-4, 14, -14)),
        regions=dict(sky=dict(exposure=-3, contrast=5, vibrance=8), water=dict(exposure=-8, vibrance=15, contrast=8, temperature=-8),
                     mountains=dict(temperature=16, exposure=3, contrast=8, clarity=7, vibrance=10),
                     trees=dict(exposure=-8, saturation=-4, temperature=8, contrast=6),
                     grass=dict(temperature=10, exposure=-3, saturation=-4), ground=dict(temperature=12, contrast=8),
                     buildings=dict(temperature=10, clarity=8)),
        curve=dict(shadows=-6, highlights=4),
        notes=["warm orange highlights, cool teal shadows (golden-hour split tone)", "deeper, richer water", "olive-warm greens"]),
    "sunset": Plan(
        basic=dict(contrast=14, highlights=-30, shadows=14, blacks=-12, dehaze=6, clarity=8, vibrance=16),
        wheels=dict(highs=(22, 42, 2), mids=(340, 10, 0), shadows=(255, 26, -2)), blending=65, balance=10,
        mixer=dict(orange=(0, 18, 4), red=(0, 10, 0), yellow=(-4, 6, 0), blue=(0, 10, -12), purple=(0, 12, -6)),
        regions=dict(sky=dict(exposure=-8, vibrance=18, contrast=10), water=dict(vibrance=10, exposure=-4),
                     mountains=dict(temperature=10, contrast=10), ground=dict(temperature=8)),
        notes=["fiery highlights, magenta mids, violet shadows (sunset)", "sky protected from clipping"]),
    "blue_hour": Plan(
        basic=dict(contrast=12, highlights=-10, shadows=8, blacks=-12, clarity=8, vibrance=12),
        wheels=dict(highs=(30, 12, 2), mids=(232, 10, 0), shadows=(225, 30, -2)), blending=60, balance=-6,
        mixer=dict(blue=(0, 10, -8), orange=(0, 8, 4)),
        regions=dict(sky=dict(exposure=-3, vibrance=10), water=dict(vibrance=8)),
        notes=["cool blue shadows with a little warm light kept"]),
    # clean midday light: neutral highlights, gentle cool shadows, deep blue sky, controlled highlights
    "sunny": Plan(
        basic=dict(contrast=8, highlights=-14, shadows=12, whites=5, blacks=-8, dehaze=6, vibrance=18, clarity=5),
        wheels=dict(highs=(50, 8, 0), shadows=(215, 14, -1)), blending=55, balance=0,
        mixer=dict(blue=(-2, 12, -12), aqua=(0, 10, -6), green=(-6, 4, -6)),
        regions=dict(sky=dict(exposure=-3, vibrance=9, contrast=5, temperature=-2), water=dict(vibrance=12, exposure=-4),
                     trees=dict(vibrance=12), grass=dict(vibrance=12)),
        notes=["deep clean blue sky", "highlight and shadow control for harsh midday contrast"]),
    "overcast": Plan(
        basic=dict(contrast=16, highlights=-8, shadows=6, whites=10, blacks=-10, clarity=14, texture=10, vibrance=22, dehaze=6),
        wheels=dict(highs=(45, 10, 2), shadows=(200, 10, 0)), blending=55,
        mixer=dict(green=(-4, 8, -4), aqua=(0, 8, 0)),
        regions=dict(sky=dict(contrast=10, exposure=-4), trees=dict(vibrance=20, clarity=6), grass=dict(vibrance=20),
                     water=dict(contrast=6, vibrance=10)),
        notes=["flat light brought to life: contrast, clarity and colour lifted"]),
    "night": Plan(
        basic=dict(contrast=10, blacks=-8, highlights=-12, clarity=6, vibrance=10),
        wheels=dict(highs=(35, 16, 0), shadows=(205, 24, -2)), blending=55,
        mixer=dict(orange=(0, 8, 4), blue=(0, 8, -6)),
        notes=["teal shadows and warm lights for night scenes"]),
    "indoor": Plan(
        basic=dict(contrast=6, highlights=-8, shadows=8, vibrance=10),
        wheels=dict(highs=(40, 8, 0)), blending=50, notes=["gentle neutral grade for indoor light"]),
}

SCENE_MODS: Dict[str, Plan] = {
    "mountain_lake": Plan(regions=dict(mountains=dict(clarity=3, dehaze=2), water=dict(vibrance=6)),
                          mixer=dict(aqua=(0, 6, -4)), notes=["mountain and water layers graded separately"]),
    "mountains": Plan(basic=dict(dehaze=6, clarity=6), regions=dict(mountains=dict(clarity=4, contrast=4)),
                      notes=["extra definition on the mountains"]),
    "coast": Plan(mixer=dict(aqua=(0, 12, -4), blue=(0, 6, -6)), regions=dict(water=dict(vibrance=10), ground=dict(temperature=6)),
                  notes=["richer aqua water, warm sand"]),
    "forest": Plan(basic=dict(dehaze=4), mixer=dict(green=(-8, 6, -10), yellow=(0, -8, 0)),
                   regions=dict(trees=dict(exposure=-4, contrast=8)), notes=["deeper, calmer greens"]),
    "countryside": Plan(mixer=dict(green=(-6, 4, -4)), regions=dict(grass=dict(vibrance=6)), notes=["lush fields"]),
    "urban": Plan(wheels=dict(shadows=(200, 8, 0), highs=(38, 8, 0)), regions=dict(buildings=dict(clarity=10, contrast=6)),
                  notes=["teal-and-orange city grade"]),
    "snow": Plan(basic=dict(whites=-6, highlights=-10, saturation=-4), wheels=dict(shadows=(215, 18, 0)), mixer=dict(blue=(0, 6, -4)),
                 notes=["snow kept clean, cool shadows, whites protected"]),
    "desert": Plan(mixer=dict(orange=(0, 8, 2), yellow=(-4, 4, 0)), wheels=dict(highs=(35, 10, 0)), notes=["warm desert tones"]),
    "general": Plan(),
}

PORTRAIT_GRADE = Plan(
    wheels=dict(highs=(45, 8, 0), shadows=(215, 8, 0)), blending=50,
    notes=["skin-safe grade: warm highlights, calm cool shadows, no colour push on skin"])


def plan(read_: SceneRead, kind: str, strength: float = 1.0) -> Plan:
    """The grade for this light and scene. Portraits get the skin-safe version with the light's mood kept gentle."""
    out = Plan()
    light = LIGHT_RECIPES[read_.light]
    scale = 0.5 + 0.5 * read_.light_conf              # unsure about the light -> a lighter hand
    if kind == "portrait":
        _merge(out, light, 0.45 * scale)
        _merge(out, PORTRAIT_GRADE, 1.0)
        out.regions = {k: v for k, v in out.regions.items() if k not in ("water", "mountains")} | \
                      {k: {kk: vv * 0.5 for kk, vv in v.items()} for k, v in out.regions.items() if k in ("water", "mountains")}
    elif kind == "people":
        _merge(out, light, 0.8 * scale)
        _merge(out, SCENE_MODS[read_.scene], 0.9)
    else:
        _merge(out, light, scale)
        _merge(out, SCENE_MODS[read_.scene], 0.5 + 0.5 * read_.scene_conf)
    out.notes.insert(0, f"{LIGHT_LABEL[read_.light]} · {SCENE_LABEL[read_.scene]}")
    return out
