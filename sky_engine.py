"""Sky studio: premium, adaptive sky treatments.

Roughly half of all landscape photos are sky, so the sky decides whether a picture looks premium. A flat "make the sky bluer"
slider cannot do that; retouchers treat the sky as a small photograph of its own:

  1. READ the sky: clear blue, scattered clouds, dramatic clouds, overcast, sunset glow, blue hour or hazy.
  2. Pick treatments that FIT that sky ("Deep blue" on a flat grey sky would only look fake, so it is not offered first).
  3. Build each treatment from the same editable pieces a retoucher would use, all confined to the sky mask:
        - a graduated darkening toward the top of the frame (adds depth, draws the eye down into the picture),
        - a CLOUD mask (brightest part of the sky): contrast, clarity, dehaze, highlight control, so clouds get texture,
        - a BLUE mask (only the blue): richer, deeper blue without touching the clouds,
        - a HORIZON glow: warm light where the sun sits.
     They appear as ordinary mask layers ("Sky: ...") in the Masks tab, so every part stays editable.
  4. Adapt the numbers to this sky (how bright it is, how much is clipped, how big it is in the frame).
A governor test-renders the result and backs off if highlights clip or the picture goes dark.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import engine
from params import EditParams, MaskLayer, LocalAdjust

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
KINDS = ["clear", "scattered", "dramatic", "overcast", "sunset", "blue_hour", "hazy"]
KIND_LABEL = {"clear": "Clear blue sky", "scattered": "Scattered clouds", "dramatic": "Dramatic clouds", "overcast": "Overcast / flat grey",
              "sunset": "Sunset / golden glow", "blue_hour": "Blue hour / dusk", "hazy": "Hazy sky"}
MIN_SKY_SHARE = 0.03
ADJ = ("exposure", "contrast", "highlights", "shadows", "vibrance", "saturation", "clarity", "dehaze", "temperature", "tint", "hue")


@dataclass
class SkyRead:
    kind: str
    confidence: float
    share: float                    # of the picture
    med: float                      # median brightness of the sky
    rng: float                      # p90 - p10 of sky brightness
    sat: float                      # mean saturation
    cloud: float                    # share of the sky that is white / grey cloud
    blue: float                     # share that is clearly blue
    warm: float                     # share that is orange / pink
    clip: float                     # share near-white
    horizon_y: float                # where the sky ends (fraction of height)
    pick: Tuple[float, float]       # a representative sky-blue pixel (fraction of width, height)
    haze_band: float                # how much paler the sky is at the horizon than at the top
    evidence: List[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"kind": self.kind, "label": KIND_LABEL[self.kind], "confidence": round(self.confidence, 2), "share": round(self.share * 100, 1),
                "brightness": round(self.med, 2), "cloud": round(self.cloud * 100), "blue": round(self.blue * 100), "warm": round(self.warm * 100),
                "evidence": self.evidence}


def read_sky(an) -> Optional[SkyRead]:
    names = an.bank.names()
    if "sky" not in names:
        return None
    size = (an.work.shape[1], an.work.shape[0])
    m = an.bank.get("sky", size)
    if m is None:
        return None
    sel = m > 0.5
    share = float(sel.mean())
    if share < MIN_SKY_SHARE:
        return None
    a = an.work
    y = a @ LUMA
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0][sel], hsv[..., 1][sel], hsv[..., 2][sel]
    ys = y[sel]
    med = float(np.median(ys))
    rng = float(np.percentile(ys, 90) - np.percentile(ys, 10))
    blue_m = (h > 190) & (h < 262) & (s > 0.22)
    warm_m = ((h < 52) | (h > 318)) & (s > 0.22)
    cloud_m = (s < 0.2) & (v > 0.55)
    blue, warm, cloud = float(blue_m.mean()), float(warm_m.mean()), float(cloud_m.mean())
    sat = float(s.mean())
    clip = float((ys > 0.97).mean())
    rows = np.nonzero(sel.any(axis=1))[0]
    horizon_y = float(np.percentile(np.nonzero(sel)[0], 97)) / a.shape[0] if len(rows) else 0.5
    # horizon vs top: a hazy sky is pale and bright near the horizon
    ysel = np.nonzero(sel)[0]
    top_cut, bot_cut = np.percentile(ysel, [25, 80])
    top = y[sel & (np.arange(a.shape[0])[:, None] <= top_cut)]
    bot = y[sel & (np.arange(a.shape[0])[:, None] >= bot_cut)]
    haze_band = float(np.median(bot) - np.median(top)) if len(top) and len(bot) else 0.0
    # a representative blue pixel (upper part of the sky, typical hue and saturation)
    ys_i, xs_i = np.nonzero(sel & (hsv[..., 0] > 190) & (hsv[..., 0] < 262) & (hsv[..., 1] > 0.22))
    if len(ys_i) > 50:
        k = int(np.argsort(hsv[..., 1][ys_i, xs_i])[len(ys_i) // 2])
        pick = (float(xs_i[k]) / a.shape[1], float(ys_i[k]) / a.shape[0])
    else:
        k = len(ysel) // 2
        pick = (0.5, float(np.percentile(ysel, 30)) / a.shape[0])

    ev: List[str] = [f"sky is {share * 100:.0f}% of the photo", f"brightness {med:.2f}, tonal range {rng:.2f}"]
    if warm > 0.32 and sat > 0.28:
        kind, conf = "sunset", min(1.0, 0.5 + warm)
        ev.append(f"{warm * 100:.0f}% of the sky is orange / pink")
    elif med < 0.30 and blue > 0.35:
        kind, conf = "blue_hour", 0.8
        ev.append("dark, deep blue")
    elif cloud > 0.62 and sat < 0.18 and rng < 0.28:
        kind, conf = "overcast", min(1.0, 0.5 + cloud * 0.5)
        ev.append(f"{cloud * 100:.0f}% grey / white cloud, little tonal variation")
    elif cloud > 0.40 and rng > 0.34:
        kind, conf = "dramatic", min(1.0, 0.5 + rng)
        ev.append(f"{cloud * 100:.0f}% cloud with strong tonal contrast")
    elif cloud > 0.14 and blue > 0.18:
        kind, conf = "scattered", 0.75
        ev.append(f"{cloud * 100:.0f}% cloud over {blue * 100:.0f}% blue")
    elif blue > 0.45:
        kind, conf = "clear", min(1.0, 0.4 + blue)
        ev.append(f"{blue * 100:.0f}% clear blue")
    elif haze_band > 0.2 and sat < 0.25:
        kind, conf = "hazy", 0.65
        ev.append("pale, bright toward the horizon")
    elif cloud > 0.4:
        kind, conf = "overcast", 0.55
    else:
        kind, conf = ("clear" if blue > cloud else "overcast"), 0.4
    return SkyRead(kind, conf, share, med, rng, sat, cloud, blue, warm, clip, horizon_y, pick, haze_band, ev)


# ------------------------------------------------------------------------------------------------ treatments
@dataclass
class Treatment:
    id: str
    label: str
    result: str
    fit: Dict[str, float]
    boost_light: Tuple[str, ...] = ()                  # lights (from the scene read) that favour this treatment


TREATMENTS: Dict[str, Treatment] = {t.id: t for t in [
    Treatment("deep_blue", "Deep polarised blue", "Rich, deep blue with clean white clouds",
              dict(clear=.95, scattered=.9, dramatic=.6, hazy=.6, overcast=.12, sunset=.3, blue_hour=.35)),
    Treatment("cloud_drama", "Cloud drama", "Textured, high-contrast clouds",
              dict(dramatic=.95, scattered=.85, clear=.4, overcast=.55, hazy=.35, sunset=.6, blue_hour=.3)),
    Treatment("golden_glow", "Golden glow", "Warm light spilling from the horizon",
              dict(sunset=.85, clear=.6, scattered=.65, dramatic=.55, overcast=.25, hazy=.6, blue_hour=.2), ("golden_hour", "sunset")),
    Treatment("sunset_fire", "Sunset fire", "Fiery orange and magenta, cool top",
              dict(sunset=.95, dramatic=.5, scattered=.4, clear=.2, overcast=.1, hazy=.3, blue_hour=.2), ("sunset", "golden_hour")),
    Treatment("overcast_rescue", "Overcast rescue", "Depth and texture for a flat grey sky, no fake colour",
              dict(overcast=.95, hazy=.7, dramatic=.3, scattered=.15, clear=.05, sunset=.1, blue_hour=.1)),
    Treatment("soft_pastel", "Soft pastel", "Light, airy, gentle colour",
              dict(clear=.6, scattered=.6, hazy=.8, overcast=.5, sunset=.5, dramatic=.2, blue_hour=.3)),
    Treatment("blue_hour", "Blue hour depth", "Cool, deep dusk with a warm ember at the horizon",
              dict(blue_hour=.95, clear=.3, sunset=.35, scattered=.2, dramatic=.2, overcast=.1, hazy=.1), ("blue_hour", "night")),
    Treatment("storm_mood", "Storm mood", "Dark, brooding, heavy sky",
              dict(dramatic=.8, overcast=.8, scattered=.45, clear=.1, sunset=.3, hazy=.2, blue_hour=.3)),
]}


def list_treatments() -> List[dict]:
    return [{"id": t.id, "label": t.label, "result": t.result} for t in TREATMENTS.values()]


def fit_scores(r: SkyRead, light: Optional[str]) -> Dict[str, float]:
    out = {}
    for t in TREATMENTS.values():
        f = t.fit.get(r.kind, 0.2)
        if light in t.boost_light:
            f = min(1.0, f + 0.12)
        out[t.id] = round(float(f), 2)
    return out


def _layer(id_: str, name: str, kind: str, adj: Dict[str, float], k: float, **kw) -> MaskLayer:
    a = LocalAdjust(**{key: float(np.clip(v * k, -100, 100)) for key, v in adj.items() if key in ADJ})
    base = dict(id=id_, name=name, kind=kind, enabled=True, limit="sky", refine=45, adj=a)
    base.update(kw)
    return MaskLayer(**base)


def _build(r: SkyRead, tid: str, k: float, an) -> Tuple[Dict[str, float], List[MaskLayer], List[str]]:
    """The sky region's own sliders plus the mask layers, for treatment `tid` at strength k, adapted to this sky."""
    bright = float(np.clip((r.med - 0.55) / 0.30, 0, 1))              # a bright sky takes (and needs) more darkening
    clip = float(np.clip(r.clip * 10, 0, 1))                           # blown parts: extra highlight recovery
    big = 0.85 if r.share > 0.40 else 1.0                              # a sky that fills the frame is toned more gently
    satroom = 1.0 - 0.6 * float(np.clip((r.sat - 0.45) / 0.35, 0, 1))  # already very saturated: leave less room
    top_y = float(np.clip(r.horizon_y, 0.25, 0.95))
    notes: List[str] = []
    grad_top = lambda exp, **kw: _layer("sky_top", "Sky: darker toward the top", "linear", dict(exposure=exp, **kw), k * big,
                                        line=[0.5, 0.0, 0.5, top_y], feather=75)
    glow = lambda **kw: _layer("sky_horizon", "Sky: horizon glow", "linear", kw, k,
                               line=[0.5, top_y, 0.5, max(0.05, top_y * 0.42)], feather=80)
    clouds = lambda **kw: _layer("sky_clouds", "Sky: clouds", "luminosity", kw, k * big,
                                 lum=[max(40.0, float(np.clip(r.med * 100 - 4, 40, 85))), 100.0], lum_soft=35)
    blue = lambda **kw: _layer("sky_blue", "Sky: blue only", "color", kw, k * big, pick=[r.pick[0], r.pick[1]], tolerance=42)
    region: Dict[str, float] = {}
    layers: List[MaskLayer] = []
    has_cloud = r.cloud > 0.10
    has_blue = r.blue > 0.15

    if tid == "deep_blue":
        region = dict(exposure=-(3 + 5 * bright), highlights=-(8 + 8 * clip), contrast=5, vibrance=8 * satroom, temperature=-2)
        layers = [grad_top(-(6 + 6 * bright))]
        if has_blue:
            layers.append(blue(saturation=12 * satroom, vibrance=8 * satroom, exposure=-5))
        if has_cloud:
            layers.append(clouds(clarity=10, contrast=8, dehaze=6, highlights=-6))
        notes.append("Blue deepened with a blue-only mask; clouds kept clean and white")
    elif tid == "cloud_drama":
        region = dict(exposure=-(6 + 6 * bright), highlights=-(14 + 10 * clip), contrast=10, clarity=8)
        layers = [grad_top(-10, contrast=6), clouds(clarity=24, contrast=16, dehaze=14, highlights=-10)]
        if has_blue:
            layers.append(blue(saturation=8 * satroom, exposure=-6))
        notes.append("Cloud-only mask: strong texture, contrast and dehaze in the clouds")
    elif tid == "golden_glow":
        region = dict(temperature=8, vibrance=12 * satroom, highlights=-14)
        layers = [grad_top(-6, temperature=-4), glow(temperature=22, exposure=4, saturation=8, vibrance=12)]
        if has_cloud:
            layers.append(clouds(clarity=8, contrast=8, highlights=-8))
        notes.append("Warm glow near the horizon, slightly cooler top for depth")
    elif tid == "sunset_fire":
        region = dict(vibrance=18 * satroom, saturation=4, contrast=10, highlights=-(24 + 8 * clip), exposure=-4)
        layers = [grad_top(-8, temperature=-6, tint=6), glow(temperature=16, saturation=10, vibrance=14)]
        if has_cloud:
            layers.append(clouds(clarity=12, contrast=10, highlights=-12))
        notes.append("Orange and magenta enriched at the horizon, cooler magenta-blue top")
    elif tid == "overcast_rescue":
        region = dict(exposure=-(2 + 4 * bright), contrast=8, clarity=6, dehaze=8, temperature=-3, highlights=-(6 + 8 * clip))
        layers = [grad_top(-14, temperature=-8, contrast=8)]
        layers.append(_layer("sky_clouds", "Sky: cloud texture", "luminosity", dict(clarity=20, dehaze=14, contrast=10), k * big,
                             lum=[25.0, 100.0], lum_soft=50))
        notes.append("Flat sky given depth (top gradient) and cloud texture; no invented colour")
    elif tid == "soft_pastel":
        region = dict(exposure=3, contrast=-6, saturation=-8, vibrance=6, highlights=-8)
        layers = [grad_top(-3), glow(temperature=6, exposure=2)]
        notes.append("Lifted, low-contrast, gently coloured")
    elif tid == "blue_hour":
        region = dict(exposure=-4, temperature=-10, vibrance=12 * satroom, contrast=8)
        layers = [grad_top(-8), glow(temperature=10, exposure=2)]
        if has_blue:
            layers.append(blue(saturation=10 * satroom, exposure=-3))
        notes.append("Cool deepened dusk with a warm ember at the horizon")
    elif tid == "storm_mood":
        region = dict(exposure=-(8 + 4 * bright), highlights=-18, contrast=16, clarity=10, saturation=-8, dehaze=10)
        layers = [grad_top(-12, saturation=-6), clouds(clarity=22, contrast=14, dehaze=12)]
        notes.append("Dark, heavy sky with textured clouds")
    region = {a: v * k * big for a, v in region.items()}
    if r.haze_band > 0.22 and r.kind in ("clear", "scattered", "hazy"):
        region["dehaze"] = region.get("dehaze", 0.0) + 6 * k
        notes.append("Pale horizon haze detected → a little dehaze added to the sky")
    return region, layers, notes


def apply(an, base: EditParams, tid: str, strength: float = 1.0) -> Tuple[EditParams, List[str], dict]:
    """The photo with sky treatment `tid` applied (strength 0..1.5). Returns (params, notes, sky summary)."""
    r = read_sky(an)
    if r is None:
        raise ValueError("No sky was found in this photo.")
    if tid not in TREATMENTS:
        raise ValueError("Unknown sky treatment.")
    k = float(np.clip(strength, 0.0, 1.5))
    t = TREATMENTS[tid]
    light = an.read.light if an.read else None
    fit = fit_scores(r, light)[tid]
    notes = [f"Sky read: {KIND_LABEL[r.kind]} ({'; '.join(r.evidence[:2])})"]
    if fit < 0.4:
        notes.append(f"'{t.label}' is not a natural fit for a {KIND_LABEL[r.kind].lower()} → applied gently")
    adapt = 0.6 + 0.4 * (fit / 0.9) if fit < 0.9 else 1.0              # a poor fit is softened automatically
    p = base.model_copy(deep=True)
    p.masks = [l for l in p.masks if not l.id.startswith("sky_")]
    ref = p.regions["sky"]
    cur_sky = {a: float(getattr(ref, a)) for a in ADJ}          # what the sky region has now (kept in proportion to 1 - strength)
    keep = 1.0 - min(k, 1.0)
    before = engine.render(an, base, 480)
    clip0 = float((np.asarray(before, np.float32).max(axis=2) / 255 >= 0.995).mean())
    for attempt in range(3):
        kk = k * adapt * (0.7 ** attempt)
        region, layers, treat_notes = _build(r, tid, kk, an)
        acc = an.reports["sky"].strength if "sky" in an.reports else 1.0
        for a in ADJ:
            setattr(ref, a, float(np.clip(region.get(a, 0.0) * acc + keep * cur_sky[a], -100, 100)))
        ref.enabled = True
        p.masks = [l for l in p.masks if not l.id.startswith("sky_")] + layers
        out = np.asarray(engine.render(an, p, 480), np.float32) / 255
        clip = float((out.max(axis=2) >= 0.995).mean()) - clip0
        med = float(np.median(out @ LUMA))
        if clip < 0.03 and med > 0.2:
            break
        notes.append("Highlights were clipping → sky edit eased" if clip >= 0.03 else "Picture came out too dark → sky edit eased")
    notes += treat_notes
    return p, notes, r.summary()


# ------------------------------------------------------------------------------------------------ suggestions for the whole photo
def suggest(an) -> dict:
    """What suits THIS photo: the sky read, ranked sky treatments, and ranked looks, each with a reason."""
    import look_engine
    out: dict = {"sky": None, "sky_treatments": [], "looks": []}
    r = read_sky(an)
    light = an.read.light if an.read else None
    if r is not None:
        fits = fit_scores(r, light)
        out["sky"] = r.summary()
        ranked = sorted(TREATMENTS.values(), key=lambda t: -fits[t.id])
        out["sky_treatments"] = [{"id": t.id, "label": t.label, "result": t.result, "fit": fits[t.id], "recommended": i < 3 and fits[t.id] >= 0.5}
                                 for i, t in enumerate(ranked)]
    facts = look_engine.read_facts(an, None, None)
    tonal = look_engine.measure_tonal(an)
    sc = {k: 0.35 for k in look_engine.LOOKS}
    why: Dict[str, str] = {}

    def add(look, v, reason):
        sc[look] = sc.get(look, 0.35) + v
        if v > 0.15 and look not in why:
            why[look] = reason

    if facts.light in ("golden_hour",):
        add("golden", 0.5, "golden-hour light")
    if facts.light in ("sunset",) or (r is not None and r.kind == "sunset"):
        add("sunset", 0.6, "warm sunset sky")
        add("golden", 0.2, "warm light")
    if facts.snowy or facts.scene in ("mountains", "mountain_lake", "snow"):
        add("alpine", 0.45, "mountains in the scene")
    if facts.present.get("water", 0) > 8:
        add("tropical", 0.3, "a lot of water")
    if r is not None and r.kind in ("overcast", "dramatic"):
        add("moody", 0.4, f"{KIND_LABEL[r.kind].lower()}")
        if r.kind == "overcast":
            add("fine_art", 0.2, "soft, flat sky")
    if r is not None and r.kind in ("clear", "scattered"):
        add("cinematic", 0.3, "clean sky with room for a graded look")
        add("natural", 0.15, "bright, clean light")
    if tonal.flat > 0.4:
        add("fine_art", 0.25, "flat, low-contrast light")
        add("cinematic", 0.15, "flat light that benefits from a grade")
    if tonal.gap > 0.5:
        add("cinematic", 0.2, "bright sky over a darker foreground")
    if an.scene.kind in ("portrait", "people"):
        add("natural", 0.35, "people in the frame")
    add("natural", 0.1, "a safe all-round edit")
    ranked_looks = sorted(sc, key=lambda k: -sc[k])
    out["looks"] = [{"id": k, "label": look_engine.LOOKS[k].label, "result": look_engine.LOOKS[k].result, "fit": round(min(1.0, sc[k]), 2),
                     "reason": why.get(k, "a good starting point")} for k in ranked_looks[:3]]
    return out
