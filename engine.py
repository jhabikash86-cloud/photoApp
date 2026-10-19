"""Non-destructive edit engine (Lightroom/Photoshop-style).

analyze(path)         expensive, once per photo: scene, region masks (scored + auto-refined), natural colour.
auto_params(an, look) the automatic edit, expressed as ordinary slider values (Lightroom 'Auto').
render(an, params)    cheap, runs on every slider move from the cached analysis. Stages, in order:
                      white balance -> basic tone -> colour balance -> local adjustments per region
                      -> people -> bokeh -> glow / vignette / grain -> overall intensity -> sharpen.
Every stage is memoised on its own inputs, so moving one slider recomputes only what depends on it.
"""
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

import adjust
import bokeh
import colorcal
import custom_masks
import denoise
import drama
import luts
import sky_replace
import depth as depthmod
import dodge_burn
import grade
import masking
import matting
import meta as metamod
import scene_grade
import masks as mk
import retouch
import scene as sc
import semseg
from adjust import Adjustments
from params import MIXER_COLORS, REGIONS, Basic, DodgeBurn, EditParams, Effects, People, Region

WORK_EDGE = 2000      # analysis + preview working resolution
PREVIEW_EDGE = 1200   # size of settled previews (1600-2000 when zoomed in); drafts are rendered at ~576
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

# Looks are modifiers on top of the automatic edit: a global scale and extra slider offsets.
LOOKS = {
    "dreamy": dict(scale=0.9, basic=dict(contrast=-8, clarity=-10, highlights=-8, shadows=8, vibrance=3),
                   effects=dict(glow=45, sharpen=25), people=dict(smooth=8, brighten=3)),
    "natural": dict(scale=1.0, basic={}, effects=dict(sharpen=35), people={}),
    "vivid": dict(scale=1.25, basic=dict(vibrance=6, saturation=4, contrast=6, clarity=4),
                  effects=dict(sharpen=45), people={}),
    "cinematic": dict(scale=1.0, basic=dict(contrast=16, saturation=-12, highlights=-10, blacks=8),
                      effects=dict(sharpen=35, vignette=-18), people={},
                      balance=dict(shadows=[-6, 0, 8], highs=[8, 0, -6])),
    "dramatic": dict(scale=1.1, basic=dict(exposure=-8, contrast=16, highlights=-14, shadows=-12, blacks=-14,
                                           vibrance=10, saturation=6, clarity=16, texture=10, dehaze=10),
                     effects=dict(sharpen=45, vignette=-16), people={},
                     regions=dict(sky=dict(exposure=-8, vibrance=12, contrast=8),
                                  grass=dict(vibrance=8), trees=dict(vibrance=8)),
                     balance=dict(shadows=[3, -5, 9], highs=[7, 0, -9])),
    "soft_portrait": dict(scale=0.9, basic=dict(clarity=-14, contrast=-6, vibrance=3),
                          effects=dict(glow=25, sharpen=20), people=dict(smooth=20, brighten=3)),
}


@dataclass
class Analysis:
    sid: str
    path: str
    full_size: tuple
    work: np.ndarray
    scene: sc.Scene
    masks: Dict[str, np.ndarray]                    # float16, work size
    reports: Dict[str, mk.MaskReport]
    wb: colorcal.WBResult
    auto: Adjustments
    stats: dict = field(default_factory=dict)
    depth: Optional[np.ndarray] = None               # float16 disparity (1 = near), work size
    bank: Optional[masking.MaskBank] = None          # all masks, with feather/refine/invert/strength + cache
    meta: Optional[metamod.Meta] = None
    coverage: Dict[str, float] = field(default_factory=dict)
    read: Optional[scene_grade.SceneRead] = None     # detected light + scene
    grade_notes: List[str] = field(default_factory=list)
    lighting_notes: List[str] = field(default_factory=list)
    sem_summary: dict = field(default_factory=dict)
    secs: float = 0.0
    created: float = field(default_factory=time.time)
    cache: dict = field(default_factory=dict)

    def summary(self) -> dict:
        s = self.scene.summary()
        s["masks"] = {k: r.summary() for k, r in self.reports.items()}
        s["natural_color"] = self.wb.summary()
        s["semantic"] = self.sem_summary
        s["lighting"] = self.lighting_notes
        if self.read:
            s["grade"] = {**self.read.summary(), "notes": self.grade_notes}
        if self.meta:
            s["meta"] = self.meta.summary()
        s["mask_names"] = self.bank.names() if self.bank else []
        s["coverage"] = self.coverage
        s["seconds"] = round(self.secs, 1)
        return s


# ---------------------------------------------------------------------------------------------
# analysis (once per photo)
# ---------------------------------------------------------------------------------------------
def _resize(a: np.ndarray, size) -> np.ndarray:
    w, h = size
    if a.shape[1] == w and a.shape[0] == h:
        return a
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA if w < a.shape[1] else cv2.INTER_CUBIC)


def _snap(work, prob, thr=(0.35, 0.4)):
    """Soft semantic probability -> edge-snapped mask (guided filter on the real image)."""
    m = np.clip((prob - thr[0]) / thr[1], 0, 1)
    return mk.guided_variant(work, m, 90, 2e-4)


MIN_SHARE = {"sky": 0.015, "trees": 0.02, "grass": 0.02, "water": 0.015, "mountains": 0.02, "buildings": 0.03,
             "ground": 0.03}


def _heuristic_masks(work, scene, mdict, reports):
    """Fallback when the segmentation model is unavailable: colour-based sky/foliage + selfie person."""
    if scene.person is not None:
        m, rep = mk.best_mask("person", work, scene.person.mask, scene.faces)
        scene.person.mask = m
        reports["person"] = rep
        mdict["person"], mdict["skin"], mdict["hair"] = m, scene.person.skin, scene.person.hair
    if scene.sky_mask is not None:
        m, rep = mk.best_mask("sky", work, scene.sky_mask)
        if scene.person is not None:
            m = m * (1 - scene.person.mask)
        scene.sky_mask = m
        reports["sky"] = rep
        mdict["sky"] = m
    excl = np.zeros(work.shape[:2], np.float32)
    for k in ("sky", "person"):
        if k in mdict:
            excl = np.maximum(excl, mdict[k].astype(np.float32))
    fol = mk.foliage_mask(work, excl)
    if (fol > 0.5).mean() > 0.02:
        m, rep = mk.best_mask("foliage", work, fol)
        mdict["trees"] = m
        reports["trees"] = rep


def _semantic_masks(work, scene, sem, mdict, reports):
    """Masks from what each pixel actually is. Every mask is still scored, and refined/retried if weak."""
    sh = sem.shares
    # A "face" the scene model says is sky/cloud or not a person is a false detection (pareidolia in clouds).
    H, W = work.shape[:2]
    def real_face(f):
        cx, cy = int(np.clip(f.pts[:, 0].mean(), 0, W - 1)), int(np.clip(f.pts[:, 1].mean(), 0, H - 1))
        return sem.groups["person"][cy, cx] > 0.3 and sem.groups["sky"][cy, cx] < 0.5
    scene.faces = [f for f in scene.faces if real_face(f)]
    if scene.person is None and sh.get("person", 0) > 0.01:
        scene.person = sc.segment_person(work)
    elif scene.person is not None and not scene.faces and sh.get("person", 0) < 0.01:
        scene.person = None
    person = None
    if scene.person is not None:
        base = np.maximum(scene.person.mask, _snap(work, sem.groups["person"], (0.4, 0.4)))
        person, rep = mk.best_mask("person", work, base, scene.faces, max_tries=4, floor=0.6)
        if not scene.faces and float((person > 0.5).mean()) < 0.008:   # a few distant specks, not a subject
            scene.person, person = None, None
        else:
            scene.person.mask = person
            reports["person"] = rep
            mdict["person"], mdict["skin"], mdict["hair"] = person, scene.person.skin, scene.person.hair
    not_person = 1 - person if person is not None else 1.0

    if not sem.indoor:   # indoors there is no sky, trees or scenery to grade
        claimed = np.zeros(work.shape[:2], np.float32)
        order = ["sky", "mountains", "water", "buildings", "trees", "grass", "ground"]
        for name in order:
            if sh.get(name, 0) < MIN_SHARE[name]:
                continue
            prob = sem.groups[name] * not_person * (1 - claimed)
            m, rep = mk.best_mask("sky" if name == "sky" else name, work, _snap(work, prob), max_tries=3, floor=0.7)
            m = m * not_person * (1 - claimed)
            if (m > 0.5).mean() < MIN_SHARE[name] * 0.6:
                continue
            claimed = np.maximum(claimed, m)
            reports[name] = rep
            mdict[name] = m

    scene.sky_mask = mdict.get("sky")
    scene.sky_frac = float((scene.sky_mask > 0.5).mean()) if scene.sky_mask is not None else 0.0
    scene.haze = 0.0 if sem.indoor else sc.estimate_haze(work, scene.sky_mask)
    biggest = scene.faces[0].area_frac if scene.faces else 0.0
    scenery = sum(sh.get(k, 0) for k in ("sky", "trees", "grass", "water", "mountains", "buildings", "ground"))
    pfrac = float((person > 0.5).mean()) if person is not None else 0.0
    if biggest >= 0.015 or pfrac >= 0.18:
        scene.kind = "portrait"          # a face, or simply a large person (e.g. in profile) is the subject
    elif biggest >= 0.001 or pfrac >= 0.02:
        scene.kind = "people"
    elif sem.indoor:
        scene.kind = "general"
    else:
        scene.kind = "landscape" if scenery >= 0.35 else "general"


def analyze(path: str, sid: Optional[str] = None) -> Analysis:
    t0 = time.time()
    work = adjust.load_float(path, WORK_EDGE)      # 16-bit for camera raw
    full = adjust.image_size(path)
    s = min(1.0, WORK_EDGE / max(work.shape[:2]))
    if s < 1:
        work = cv2.resize(work, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    scene = sc.analyze_scene(work)
    reports: Dict[str, mk.MaskReport] = {}
    mdict: Dict[str, np.ndarray] = {}
    sem = None
    if scene.kind != "screenshot":
        sem = semseg.run(work)
        if sem is not None:
            _semantic_masks(work, scene, sem, mdict, reports)
        else:
            _heuristic_masks(work, scene, mdict, reports)

    if scene.kind != "screenshot":
        wb = colorcal.estimate(work, mdict.get("sky"), mdict.get("skin"), mdict.get("trees"))
        auto = adjust.analyze(adjust.to_image(colorcal.apply_gains(work, wb.gains)))
    else:
        wb, auto = colorcal.WBResult(), Adjustments()

    stats = {}
    try:
        stats["noise"] = denoise.estimate_sigma((np.clip(work, 0, 1) * 255 + 0.5).astype(np.uint8))
    except Exception:
        pass
    sky = mdict.get("sky")
    if sky is not None and (sky > 0.5).sum() > 100:
        hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)
        stats["sky_sat"] = float(hsv[..., 1][sky > 0.5].mean())
    sm = mdict.get("skin")
    if sm is not None and (sm > 0.6).sum() > 200:
        stats["skin_luma"] = float(np.median((work @ LUMA)[sm > 0.6]))
    yy = work @ LUMA
    stats["dark_share"] = float((yy < 0.18).mean())       # how much of the frame is deep shadow / dark backdrop
    pm = mdict.get("person")
    if pm is not None and (pm > 0.6).any():
        stats["person_luma"] = float(np.median((work @ LUMA)[pm > 0.6]))
        stats["person_frac"] = float((pm > 0.5).mean())

    dmap = None
    if scene.kind != "screenshot":      # real depth: bokeh for portraits, and the foreground of every scene
        d = depthmod.estimate(work)
        dmap = d.astype(np.float16) if d is not None else None
    an = Analysis(sid=sid or uuid.uuid4().hex, path=path, full_size=full, work=work, scene=scene,
                  masks={k: v.astype(np.float16) for k, v in mdict.items()}, reports=reports, wb=wb, auto=auto,
                  stats=stats, depth=dmap, sem_summary=sem.summary() if sem else {})
    if scene.faces and "skin" in an.masks:      # face region (landmark oval): the shape dodge & burn sculpts
        fm = np.zeros(work.shape[:2], np.uint8)
        for f in scene.faces:
            cv2.fillConvexPoly(fm, cv2.convexHull(f.pts[sc.OVAL].astype(np.int32)), 255)
        fm = cv2.GaussianBlur(fm.astype(np.float32) / 255.0, (0, 0), 1.5)
        an.masks["face"] = (fm * (an.masks["skin"].astype(np.float32) > 0.3)).astype(np.float16)
    an.bank = masking.MaskBank(an.masks, an.depth)
    an.coverage = an.bank.coverage()
    if scene.kind != "screenshot":
        an.meta = metamod.read(path)
        an.read = scene_grade.read(work, mdict.get("sky"), (sem.summary()["shares"] if sem else {}),
                                   bool(sem and sem.indoor), scene.kind, an.meta, mdict)
    an.secs = time.time() - t0
    return an


# ---------------------------------------------------------------------------------------------
# the automatic edit, as slider values
# ---------------------------------------------------------------------------------------------
NOT_SCALED = {"enabled", "mask_strength", "feather", "refine", "invert", "strength", "keep_whites"}   # controls, not adjustments


def _scale_model(model, k):
    """Scale a model's adjustment sliders by k (mask controls and switches are left alone)."""
    for name in type(model).model_fields:
        v = getattr(model, name)
        if name not in NOT_SCALED and isinstance(v, (int, float)) and not isinstance(v, bool):
            setattr(model, name, v * k)


def _clamp(model):
    for name, f in type(model).model_fields.items():
        v = getattr(model, name)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            lo = next((m.ge for m in f.metadata if hasattr(m, "ge")), -100)
            hi = next((m.le for m in f.metadata if hasattr(m, "le")), 100)
            setattr(model, name, float(min(hi, max(lo, v))))


# How much the *average* saturation may rise under Auto, per look (Dramatic and Vivid are meant to be rich).
SAT_CAP = {"natural": 0.13, "dreamy": 0.12, "soft_portrait": 0.11, "cinematic": 0.13, "vivid": 0.18, "dramatic": 0.20}
CLIP_CAP = 0.02      # Auto may not clip more than 2% extra pixels


def _picture_stats(a: np.ndarray) -> tuple:
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    return float(hsv[..., 1].mean()), float((a.max(axis=2) >= 0.99).mean())


def _govern(an: Analysis, p: EditParams) -> EditParams:
    """Safety governor: test-render the automatic edit small, measure what it did to the picture and scale the
    colour boosts back if saturation or clipping rose past the limits. Auto should never mess a picture up."""
    if an.scene.kind == "screenshot":
        return p
    src, size, _ = _target(an, 480)
    sat0, clip0 = _picture_stats(src)
    skin_m = _masks_at(an, size).get("skin")
    skin_sel = skin_m > 0.6 if skin_m is not None else None
    if skin_sel is not None and skin_sel.sum() < 200:
        skin_sel = None

    def skin_chroma(a):
        lab = cv2.cvtColor(a[skin_sel].reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)
        return float(np.hypot(*np.median(lab[:, 1:], axis=0)))

    c_skin0 = skin_chroma(src) if skin_sel is not None else 0.0
    ys0 = src @ LUMA
    luma0, black0 = float(ys0.mean()), float(np.percentile(ys0, 5))
    for _ in range(4):
        out = np.asarray(render(an, p, 480), np.float32) / 255
        sat1, clip1 = _picture_stats(out)
        ys1 = out @ LUMA
        dsat, dclip = sat1 - sat0, clip1 - clip0
        dluma, dblack = float(ys1.mean()) - luma0, float(np.percentile(ys1, 5)) - black0
        cap = SAT_CAP.get(p.look, 0.12)
        skin_up = skin_chroma(out) / max(c_skin0, 1.0) - 1 if skin_sel is not None and c_skin0 > 3 else 0.0
        too_bright = dluma > 0.06            # Auto must not wash a picture out ...
        hazy = dblack > 0.025                # ... or lift its blacks into a grey veil
        if dsat <= cap and dclip <= CLIP_CAP and skin_up <= 0.08 and not too_bright and not hazy:
            break
        k = 1.0
        if dsat > cap:
            k = min(k, max(0.4, cap / dsat))
        if dclip > CLIP_CAP:
            k = min(k, 0.75)
        if skin_up > 0.08:        # skin must not get more colourful (that is what looks orange)
            k = min(k, 0.7)
            p.people.warmth = min(p.people.warmth, 0)
            p.people.tone_fix = min(100, p.people.tone_fix + 15)
        for obj in [p.basic] + list(p.regions.values()):
            for name in ("vibrance", "saturation"):
                v = getattr(obj, name)
                if v > 0:
                    setattr(obj, name, v * k)
        if dclip > CLIP_CAP and p.basic.exposure > 0:
            p.basic.exposure *= 0.5
        if too_bright or hazy:
            f = 0.55
            p.basic.exposure = min(p.basic.exposure, 0.0) if too_bright else p.basic.exposure * f
            p.basic.shadows *= f
            p.dodge_burn.dodge *= f
            p.dodge_burn.separation *= 0.8
            p.effects.glow *= f
            p.people.brighten *= f
            p.people.lift *= f
            if hazy:
                p.basic.blacks = min(p.basic.blacks, 0.0) - 6.0
                p.grading.glob.lum = min(p.grading.glob.lum, 0.0)
    return p


# Auto stacks several sources (measured correction + scene recipe + light recipe). Each is sensible alone; added together they
# can reach extremes (highlights -70, shadows +66) that flatten the picture into the crunchy HDR look. These are the most
# Auto will ever apply; the user can still go further by hand.
AUTO_LIMITS = dict(exposure=(-30, 30), contrast=(-10, 28), highlights=(-42, 10), shadows=(-10, 38), whites=(-20, 15),
                   blacks=(-22, 8), dehaze=(-10, 24), clarity=(-20, 12), texture=(-20, 10), vibrance=(-40, 40), saturation=(-40, 14))


def _limit_tone(p: EditParams):
    for k, (lo, hi) in AUTO_LIMITS.items():
        setattr(p.basic, k, float(min(hi, max(lo, getattr(p.basic, k)))))
    if p.basic.exposure > 8:      # do not lift the same dark areas twice (exposure AND shadows): it makes grey, noisy mud
        p.basic.shadows = float(min(p.basic.shadows, 38 - 0.9 * p.basic.exposure))
    for r in p.regions.values():
        r.clarity = float(min(10, max(-20, r.clarity)))
        r.dehaze = float(min(14, max(-10, r.dehaze)))
        r.contrast = float(min(16, max(-20, r.contrast)))
        r.exposure = float(min(14, max(-18, r.exposure)))


GRADE_SCALE = {"natural": 0.8, "vivid": 1.1, "dramatic": 1.3, "cinematic": 1.1, "dreamy": 0.7, "soft_portrait": 0.6}


def _apply_plan(an: Analysis, p: EditParams, read_: scene_grade.SceneRead, kind: str, g: float):
    """Write the scene/light grade into ordinary slider values (so every part stays editable)."""
    plan = scene_grade.plan(read_, kind)
    for k, v in plan.basic.items():
        setattr(p.basic, k, getattr(p.basic, k) + v * g)
    for name, (h, s_, l) in plan.wheels.items():
        w = getattr(p.grading, "glob" if name == "glob" else name)
        w.hue, w.sat, w.lum = h % 360, min(100, s_ * g), l * g
    if plan.blending is not None:
        p.grading.blending = plan.blending
    if plan.balance is not None:
        p.grading.balance = plan.balance * g
    for color, (h, s_, l) in plan.mixer.items():
        if color in p.mixer:
            p.mixer[color] = [p.mixer[color][0] + h * g, p.mixer[color][1] + s_ * g, p.mixer[color][2] + l * g]
    for rname, d in plan.regions.items():
        if rname in p.regions and an.bank.has(rname):
            for k, v in d.items():
                setattr(p.regions[rname], k, getattr(p.regions[rname], k) + v * g)
    for k, v in plan.curve.items():
        setattr(p.curve, k, getattr(p.curve, k) + v * g)
    for mdl in [p.basic, p.curve] + list(p.regions.values()):
        _clamp(mdl)
    for w in (p.grading.shadows, p.grading.mids, p.grading.highs, p.grading.glob):
        w.sat = float(min(100, max(0, w.sat))); w.lum = float(min(100, max(-100, w.lum)))
    p.mixer = {c: [float(min(100, max(-100, x))) for x in v] for c, v in p.mixer.items()}
    an.grade_notes = plan.notes


def auto_lighting(an: Analysis) -> DodgeBurn:
    """The automatic dodge & burn proposal for this photo (also stores the plain-language reasons)."""
    if an.scene.kind == "screenshot":
        return DodgeBurn()
    size = (an.work.shape[1], an.work.shape[0])
    masks = {n: an.bank.get(n, size) for n in ("skin", "face", "person", "foreground", "background")}
    arr = colorcal.apply_gains(an.work, an.wb.gains)
    res, notes = dodge_burn.auto_lighting(arr, {k: v for k, v in masks.items() if v is not None})
    an.lighting_notes = notes
    return res


def pick_look(an: Analysis) -> str:
    """The look 'Auto' chooses for this photo."""
    return "dreamy" if an.scene.kind in ("portrait", "people") else "natural"


def auto_params(an: Analysis, look: str = "dreamy", light: Optional[str] = None, scene: Optional[str] = None) -> EditParams:
    """What 'Auto' would do for this photo, as ordinary Lightroom-style slider values."""
    if look == "auto":
        look = pick_look(an)
    cfg = LOOKS[look]
    kind = an.scene.kind
    p = EditParams(look=look, wb_mode="auto")     # Auto uses the natural-colour estimate; a neutral EditParams() does not
    if kind == "screenshot":
        return p
    a = an.auto
    haze = an.scene.haze
    indoor = bool(an.sem_summary.get("indoor"))

    # ---- global tone: measured corrections + a scene recipe -------------------------------------
    b = p.basic
    b.exposure = float(np.clip(a.exposure / 3 * 100 * 0.8, -25, 25))
    b.contrast = float(np.clip(a.contrast * 0.6, 0, 14))
    b.highlights = float(np.clip(a.highlights * 1.2, -40, 0))
    b.shadows = float(np.clip(a.shadows * 0.9, 0, 40))
    if kind == "landscape":
        b.contrast += 8; b.highlights -= 12; b.shadows += 10; b.whites += 5; b.blacks -= 8
        b.dehaze = min(32, 6 + 28 * haze)
        b.vibrance, b.saturation, b.clarity, b.texture = 20, 0, 7, 5
    elif kind == "general":
        b.contrast += 6; b.shadows += 8; b.vibrance, b.clarity, b.texture = 20, 6, 5
        b.dehaze = 0 if indoor else 3 + 12 * haze
    else:   # portrait / people
        b.contrast += 6; b.highlights -= 8; b.shadows += 8
        b.vibrance, b.clarity, b.texture = 18, 3, 0
        b.dehaze = 2 + 14 * haze * (1 if kind == "people" else 0.3)
    # Low-key portrait (dark backdrop on purpose, subject well lit): the frame's median is low because of the mood,
    # not because the picture is under-exposed. Expose for the skin and keep the shadows deep, as a retoucher would.
    sk = an.stats.get("skin_luma")
    lowkey = kind in ("portrait", "people") and sk is not None and an.stats.get("dark_share", 0) > 0.28 and sk > 0.38
    if lowkey:
        b.exposure = float(np.clip((0.50 - sk) * 60, -8, 14))
        b.shadows = min(b.shadows, 4.0)
        b.highlights = max(b.highlights, -14.0)
        b.blacks, b.contrast = -10.0, max(b.contrast, 12.0)
        b.dehaze = 0
    _scale_model(b, cfg["scale"])
    for k, v in cfg["basic"].items():
        setattr(b, k, getattr(b, k) + v)
    if lowkey:                       # a look must not undo the low-key mood either
        b.shadows = min(b.shadows, 6.0)
        b.exposure = min(b.exposure, 14.0)

    # ---- local regions --------------------------------------------------------------------------
    scenery = {"portrait": 0.6, "people": 0.85}.get(kind, 1.0) * cfg["scale"]
    R = p.regions
    sat_sky = an.stats.get("sky_sat", 0.3)
    overcast = float(np.clip((0.22 - sat_sky) / 0.15, 0, 1))      # pale / grey sky: gentler
    if "sky" in an.masks:
        R["sky"] = Region(exposure=-3 * (1 - 0.4 * overcast), contrast=6, vibrance=18 * (1 - 0.65 * overcast),
                          saturation=0, temperature=-2)
    if "mountains" in an.masks:
        R["mountains"] = Region(contrast=6, clarity=5, dehaze=3 + 10 * haze, vibrance=10, temperature=-2)
    if "trees" in an.masks:
        R["trees"] = Region(exposure=1, contrast=4, vibrance=18, clarity=4)
    if "grass" in an.masks:
        R["grass"] = Region(exposure=2, contrast=3, vibrance=20, clarity=3, hue=3)
    if "water" in an.masks:
        R["water"] = Region(contrast=6, vibrance=20, clarity=4, dehaze=3)
    if "buildings" in an.masks:
        R["buildings"] = Region(contrast=6, clarity=8, vibrance=6, dehaze=2)
    if "ground" in an.masks:
        R["ground"] = Region(contrast=5, clarity=5, vibrance=10, temperature=2)
    for name, mods in cfg.get("regions", {}).items():
        if name in an.masks:
            for k, v in mods.items():
                setattr(R[name], k, getattr(R[name], k) + v / max(scenery, 0.3))
    for name, r in R.items():
        _scale_model(r, scenery * (an.reports[name].strength if name in an.reports else 1.0))
        _clamp(r)

    # ---- people ---------------------------------------------------------------------------------
    if "person" in an.masks:
        pe = p.people
        st = cfg["scale"] * (an.reports["person"].strength if "person" in an.reports else 1.0)
        pe.smooth, pe.tone_fix, pe.eyes, pe.lips, pe.sculpt = 38 * st, 70, 28 * st, 14 * st, 30 * st
        if kind == "portrait" and an.stats.get("person_frac", 0) > 0.15:
            pe.eyes = 42 * st                       # eyes are the focus of a close portrait
        pe.brighten = 10 * st
        luma = an.stats.get("person_luma", 0.45)
        pe.lift = float(np.clip((0.42 - luma) / 0.25, 0, 1) * 70)
        for k, v in cfg["people"].items():
            setattr(pe, k, getattr(pe, k) + v)
        _clamp(pe)

    # ---- effects / looks ------------------------------------------------------------------------
    e = p.effects
    for k, v in cfg["effects"].items():
        setattr(e, k, v)
    if "noise" in an.stats:
        e.denoise, e.denoise_color = denoise.auto_amounts(an.stats["noise"])
    if lowkey:
        e.glow = min(e.glow, 12)
    if kind not in ("portrait", "people"):
        e.glow = min(e.glow, 20) if look == "dreamy" else 0
    pf = an.stats.get("person_frac", 0)
    if kind == "portrait" and "person" in an.masks and pf > 0.06:
        e.bokeh = 30      # professional but natural: subtle, distance-based
        e.vignette = e.vignette or -12
    elif kind == "people" and "person" in an.masks and 0.10 <= pf <= 0.7 and look in ("dreamy", "soft_portrait", "natural"):
        e.bokeh = 18      # a hint of portrait-mode separation for environmental portraits
    for name, vals in cfg.get("balance", {}).items():
        setattr(p, f"balance_{'highs' if name == 'highs' else name}", list(vals))
    _clamp(b); _clamp(e)
    if an.read is not None:
        r = an.read
        if light in scene_grade.LIGHTS or scene in scene_grade.SCENES:   # the user chose the light and/or the scene
            import copy
            r = copy.copy(an.read)
            if light in scene_grade.LIGHTS:
                r.light, r.light_conf = light, 1.0
            if scene in scene_grade.SCENES:
                r.scene, r.scene_conf = scene, 1.0
        _apply_plan(an, p, r, kind, GRADE_SCALE.get(look, 1.0))
    _limit_tone(p)
    p.dodge_burn = auto_lighting(an)
    return _govern(an, p)


# ---------------------------------------------------------------------------------------------
# render (every slider move)
# ---------------------------------------------------------------------------------------------
def _target(an: Analysis, edge: Optional[int]):
    """(src array, (w,h), scale vs work). edge=None means the full-resolution original."""
    W, H = an.work.shape[1], an.work.shape[0]
    if edge is None:
        src = adjust.load_float(an.path)
    else:
        s = min(1.0, min(edge, WORK_EDGE) / max(W, H))
        src = _resize(an.work, (int(W * s), int(H * s)))
    return src, (src.shape[1], src.shape[0]), src.shape[1] / W


def _nomemo(an, key, fn):
    return fn()


def _masks_at(an: Analysis, size, keep: bool = True) -> Dict[str, np.ndarray]:
    """Every mask (including derived foreground/background) at `size`, from the shared mask bank."""
    if not keep:
        return {n: m for n in an.bank.names() + ["hair"] if (m := an.bank.get(n, size, cache=False)) is not None}
    return _memo(an, ("masks", size), lambda: {n: an.bank.get(n, size) for n in an.bank.names() + ["hair"]
                                                if an.bank.get(n, size) is not None})


def region_mask(an: Analysis, name: str, size, src: np.ndarray, r, base: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """An automatic region as the user wants it: the detected mask, corrected by any paint strokes (add area / remove
    area), then shaped by the region's own invert, feather, edge-refine and strength controls."""
    if base is None:
        base = an.bank.get(name, size)
    if base is None:
        return None
    if getattr(r, "strokes", None):
        add = [st for st in r.strokes if not st.erase]
        rem = [st for st in r.strokes if st.erase]
        m = base
        if add:
            m = np.maximum(m, custom_masks.brush(size, add))
        if rem:
            m = m * (1.0 - custom_masks.brush(size, [st.model_copy(update={"erase": False}) for st in rem]))
        base = m
    return an.bank.shape(base, src, r.mask_strength, r.feather, r.refine, r.invert)


def _faces_at(an: Analysis, scale: float) -> List[sc.Face]:
    return [sc.Face(pts=f.pts * scale, area_frac=f.area_frac, width=f.width * scale,
                    focus_score=f.focus_score, focus=f.focus) for f in an.scene.faces]


def _memo(an: Analysis, key, fn):
    if key in an.cache:
        return an.cache[key]
    if len(an.cache) > 36:      # entries are preview-size arrays or masks: keep memory bounded
        for k in list(an.cache)[:16]:
            del an.cache[k]
    an.cache[key] = fn()
    return an.cache[key]


def _js(obj) -> str:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    elif isinstance(obj, dict):
        obj = {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in obj.items()}
    return json.dumps(obj, sort_keys=True)


def _wb_gains(an: Analysis, p: EditParams) -> np.ndarray:
    return an.wb.candidates.get(p.wb_mode, an.wb.candidates.get("auto", np.ones(3, np.float32)))


def _sky_gradient(mask: np.ndarray) -> np.ndarray:
    """1 at the top of the frame fading to ~0.3 at the lowest sky pixel (the horizon)."""
    H = mask.shape[0]
    rows = mask.mean(axis=1)
    bottom = int(np.max(np.nonzero(rows > 0.1)[0])) + 1 if (rows > 0.1).any() else H
    g = np.clip(1 - np.arange(H, dtype=np.float32) / max(bottom, 1), 0, 1) ** 0.8
    return (0.3 + 0.7 * g)[:, None]


def _people(arr, an, m, faces, pe: People):
    """People: face retouch (distance-scaled), body-skin tone, warmth, backlight lift, whole-person exposure."""
    pm = m.get("person")
    out = arr.copy()
    if pm is None:
        return out
    rs = an.reports["person"].strength if "person" in an.reports else 1.0
    f = lambda v: v / 100 * rs
    pmasks: Dict[str, np.ndarray] = {}
    for face in faces:
        tier = 0.25 + 0.75 * float(np.clip((face.area_frac - 0.001) / 0.014, 0, 1))
        blur_pen = 0.7 if face.focus == "blurry" else 1.0
        retouch.retouch_face(out, face, 1.0, pmasks, 1.0, pe.brighten / 100,
                             smooth=f(pe.smooth) * tier * blur_pen, eyes_amt=f(pe.eyes) * tier,
                             tone_amt=f(pe.tone_fix), sculpt_amt=f(pe.sculpt) * tier, lips_amt=f(pe.lips) * tier)
    skin = m.get("skin", np.zeros_like(pm))
    face_skin = np.clip(pmasks.get("skin", np.zeros_like(pm)) * 3, 0, 1)
    face_skin = cv2.GaussianBlur(face_skin, (0, 0), max(2.0, max(out.shape[:2]) / 400))
    body = np.clip(skin * (1 - face_skin), 0, 1)
    retouch.tone_body_skin(out, body, f(pe.tone_fix), pe.brighten / 100 * 0.7)

    if pe.warmth:   # skin warmth, masked to skin only
        sk = np.clip(np.maximum(skin, pmasks.get("skin", 0)), 0, 1)[..., None]
        out = out + sk * (grade.temp_tint(out, pe.warmth * 0.6) - out)
    if pe.lift:     # backlit people (sunsets) come out dark: lift the person alone
        sel = pm > 0.6
        luma = out @ LUMA
        med = float(np.median(luma[sel])) if sel.any() else 0.4
        gain = 1 + (float(np.clip(0.40 / max(med, 0.04), 1.0, 1.8)) - 1) * (pe.lift / 100) * rs
        out = np.clip(out * (1 + (gain - 1) * pm[..., None]), 0, 1)
    if pe.exposure:
        out = out + pm[..., None] * (grade.exposure(out, pe.exposure) - out)
    return np.clip(out, 0, 1)


def render(an: Analysis, p: EditParams, edge: Optional[int] = PREVIEW_EDGE, draft: bool = False) -> adjust.Image.Image:
    """Render the photo for the given sliders. edge=None renders at full resolution.
    draft=True is the fast path used while a slider is being dragged: it skips the slow finishing steps
    (noise reduction, background blur, glow, grain, sharpening) so the tonal and colour effect shows at once."""
    if edge is not None:
        key = ("final", edge, draft, _js(p))
        if key in an.cache:
            return an.cache[key]
        img = _render(an, p, edge, draft)
        _memo(an, key, lambda: img)
        return img
    return _render(an, p, edge, draft)


def _render(an: Analysis, p: EditParams, edge: Optional[int], draft: bool) -> adjust.Image.Image:
    if p.look not in LOOKS:
        raise ValueError(f"Unknown look '{p.look}'. Choose from: {', '.join(LOOKS)}")
    src, size, scale = _target(an, edge)
    full_res = edge is None
    _memo = _nomemo if full_res else globals()['_memo']      # a one-off full-size export must not park gigabytes in the session cache
    if an.scene.kind == "screenshot":
        return adjust.to_image(src)
    m = dict(_masks_at(an, size, keep=not full_res))
    src0 = src                                                     # the untouched original: 'Edit strength' blends back to this
    sr = p.sky_replace
    sky_override = None
    if sr.id and sr.blend > 0 and "sky" in m:
        rep = _memo(an, ("skyrep", _js(sr), size), lambda: sky_replace.apply(an, src, m, sr))
        if rep[1] is not None:
            src, sky_override = rep
            m["sky"] = sky_override                               # later stages follow the cleaned sky edge
    faces = _faces_at(an, scale)
    ck = (edge, size) + ((_js(sr),) if sky_override is not None else ())     # every cached stage depends on which sky is in the picture
    matte = None
    if "person" in m:   # a proper matte at THIS resolution: clean edges, hair-aware, no speckle
        matte = _memo(an, ("matte", size) + ck[2:], lambda: matting.refine_alpha(src, m["person"], m.get("hair")))
        m["person"] = matte[0]

    # 1. white balance
    wb_key = (p.wb_mode, round(p.basic.temperature, 1), round(p.basic.tint, 1))
    arr_wb = _memo(an, ("wb",) + ck + wb_key, lambda: colorcal.apply_gains(
        src, _wb_gains(an, p), p.basic.temperature, p.basic.tint))

    # 1b. noise reduction first, so shadow lifts and clarity don't amplify noise
    e = p.effects
    if (e.denoise > 0 or e.denoise_color > 0) and not draft:
        arr_wb = _memo(an, ("nr", round(e.denoise), round(e.denoise_color)) + ck + wb_key,
                       lambda: denoise.apply(arr_wb, e.denoise, e.denoise_color))
        wb_key = wb_key + (round(e.denoise), round(e.denoise_color))

    # 2. basic tone (global)
    bdict = p.basic.model_dump()
    bdict.pop("temperature"); bdict.pop("tint")
    mixer = p.mixer if p.grading_enabled else None
    basic_key = ck + wb_key + (json.dumps(bdict, sort_keys=True), _js(p.mixer) if p.grading_enabled else "off")
    skin_protect = m.get("skin")
    arr_g, ctx_g = _memo(an, ("basic",) + basic_key,
                         lambda: grade.apply_basic(arr_wb, p.basic, mixer, skin_protect, m.get("sky")))
    out = arr_g
    if p.grading_enabled:
        out = colorcal.color_balance(out, p.balance_shadows, p.balance_mids, p.balance_highs)
        out = grade.parametric_curve(out, p.curve)
    out = grade.point_curve(out, p.tone)
    cb = out

    # 3. local adjustments per region: the same shared masks, shaped by each adjustment's own mask controls
    mask_keys = []
    for name in REGIONS:
        r = p.regions.get(name)
        if r is None or not r.enabled or not an.bank.has(name):
            continue
        strength = an.reports[name].strength if name in an.reports else 1.0
        eff = _memo(an, ("rmask", name, _js(r.model_dump(include={'strokes', 'mask_strength', 'feather', 'refine', 'invert'})), _js(sr) if (name == "sky" and sky_override is not None) else "") + ck,
                    lambda: region_mask(an, name, size, src, r, sky_override if name == "sky" else None))
        if eff is None:
            continue
        if name == "sky" and not r.invert:   # a real sky deepens toward the top and stays light near the horizon
            eff = eff * _sky_gradient(eff)
        eff = eff * strength
        delta = _memo(an, ("region", name, _js(r), strength) + basic_key,
                      lambda: grade.region_delta(arr_g, ctx_g, r, eff))
        mask_keys.append((name, _js(r)))
        if delta is not None:
            (y0, y1, x0, x1), patch = delta
            if out is arr_g or out is cb:
                out = out.copy()
            out[y0:y1, x0:x1] += patch
    # 3b. the user's own masks: gradients, radial, brush, luminosity range, colour range
    layer_keys = []
    for layer in p.masks:
        if not layer.enabled:
            continue
        a = layer.adj
        if not any(getattr(a, f) for f in type(a).model_fields):
            continue
        lkey = ("custom", _js(layer)) + ck
        mk_ = _memo(an, lkey + ("mask",), lambda: custom_masks.build(layer, size, src, an.bank))
        if mk_ is None:
            continue
        delta = _memo(an, lkey + ("delta",) + basic_key, lambda: grade.region_delta(arr_g, ctx_g, a, mk_))
        layer_keys.append(_js(layer))
        if delta is not None:
            (y0, y1, x0, x1), patch = delta
            if out is arr_g or out is cb:
                out = out.copy()
            out[y0:y1, x0:x1] += patch
    out = np.clip(out, 0, 1)
    if full_res:       # the pre-local-adjustment copies and the shared context fields are finished with: free ~2 GB at 48 MP
        del arr_g, ctx_g, cb, arr_wb

    if p.grading_enabled:
        out = grade.apply_wheels(out, p.grading, m.get("skin"), m.get("sky"))

    up = basic_key + (json.dumps(mask_keys), json.dumps(layer_keys), _js(p.tone), _js(p.grading), _js(p.lut), json.dumps([p.balance_shadows, p.balance_mids, p.balance_highs]),
                      _js(p.curve), p.grading_enabled)

    # 3b'. Drama: mood colour over all tones, sky burn that keeps the clouds, bloom and mist
    if p.drama.enabled and p.drama.amount > 0:
        dmap_d = None
        if an.depth is not None:
            dmap_d = _memo(an, ("depth", size) + ck[2:], lambda: _resize(an.depth.astype(np.float32), size).clip(0, 1))
        dsrc = out
        out = _memo(an, ("drama", _js(p.drama), hash(up)) + ck, lambda: drama.apply(dsrc, m, dmap_d, p.drama))
        up = up + (_js(p.drama),)

    # 3c. creative LUT (an imported .cube or a built-in look), after the grade and before the finishing steps
    if p.lut.id and p.lut.intensity > 0:
        lut = luts.get(p.lut.id)
        if lut is not None:
            lsrc = out
            out = _memo(an, ("lut", p.lut.id, round(p.lut.intensity), hash(up)) + ck, lambda: luts.apply(lsrc, lut, p.lut.intensity))

    # 4. people
    if "person" in m and p.people.enabled:
        out = _memo(an, ("people", _js(p.people), hash(up)) + ck, lambda: _people(out, an, m, faces, p.people))
        up = up + (_js(p.people),)

    # 4b. dodge & burn: smooth, mask-aware lighting (after retouching, before the background blur)
    db = p.dodge_burn
    if db.enabled and any((db.dodge, db.burn, db.separation, db.form)) and db.strength > 0:
        out = _memo(an, ("dodgeburn", _js(db), hash(up)) + ck, lambda: dodge_burn.apply(out, m, db))
        up = up + (_js(db),)

    # 5. bokeh (blurs everything except the person; far scenery is fully blurred)
    if e.bokeh > 0 and "person" in m and not draft:
        far = None
        for k in ("sky", "mountains"):
            if k in m:
                far = m[k] if far is None else np.maximum(far, m[k])
        dmap = None
        if an.depth is not None:
            dmap = _memo(an, ("depth", size), lambda: _resize(an.depth.astype(np.float32), size).clip(0, 1))
        bsrc = out
        out = _memo(an, ("bokeh", hash(up), e.bokeh, e.bokeh_size, e.bokeh_highlights, e.bokeh_focus) + ck,
                    lambda: bokeh.apply(bsrc, m["person"], e.bokeh, e.bokeh_size, e.bokeh_highlights, far, dmap, e.bokeh_focus,
                                        m.get("hair"), float(an.stats.get("noise", 0.0)), matte))
        up = up + ("bokeh", e.bokeh, e.bokeh_size, e.bokeh_highlights, e.bokeh_focus)

    # 5b. keep the picture's own white level: highlight recovery and sky burn squeeze the top of the tonal range (a pro edit keeps real whites)
    if p.basic.keep_whites > 0:
        out = _keep_whites(out, src0, p.basic.keep_whites / 100.0)

    # 6. finishing
    if e.glow and not draft:
        out = retouch.dreamy_glow(out, e.glow / 100 * 0.3, m.get("sky"), m.get("skin"))
    out = grade.vignette(out, e.vignette, e.vignette_mid)
    if not draft:
        out = grade.grain(out, e.grain)

    result = np.clip(src0 + (np.clip(out, 0, 1) - src0) * p.intensity, 0, 1)
    img = adjust.to_image(result)
    return img if draft else adjust.sharpen_image(img, e.sharpen * min(1.0, p.intensity))


def _keep_whites(out: np.ndarray, src0: np.ndarray, amount: float) -> np.ndarray:
    """Lift the upper tones (hue kept) until the brightest ~0.5% is back at the ORIGINAL's level, never above 0.97; shadows and mids are untouched.
    Acts only when the whole top end was pulled down; a picture that still has its bright light is left alone."""
    LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
    target = min(0.97, float(np.percentile(src0[::4, ::4] @ LUMA, 99.5)))
    if target < 0.6:
        return out
    hi = float(np.percentile(out[::4, ::4] @ LUMA, 99.5))
    if hi >= target - 0.02:
        return out
    g = 1.0 + (target / max(hi, 0.25) - 1.0) * amount
    y = out @ LUMA
    w = np.clip((y - 0.30) / 0.45, 0, 1)
    w = w * w * (3 - 2 * w)
    return np.clip(out * (1.0 + (g - 1.0) * w)[..., None], 0, 1).astype(np.float32)


def regions_overlay(an: Analysis, edge: int = 1400) -> adjust.Image.Image:
    """What the engine detected, one colour per region; outline = faces."""
    src, size, scale = _target(an, edge)
    m = _masks_at(an, size)
    o = src.copy()
    colors = {"person": (1.0, 0.55, 0.0), "sky": (0.1, 0.4, 1.0), "mountains": (0.55, 0.4, 0.3), "trees": (0.1, 0.65, 0.1),
              "grass": (0.5, 0.95, 0.15), "water": (0.0, 0.9, 0.9), "buildings": (0.7, 0.25, 0.9),
              "ground": (0.85, 0.7, 0.3), "skin": (0.1, 0.9, 0.3), "other": (1.0, 0.1, 0.6)}
    for name in ("sky", "mountains", "water", "buildings", "trees", "grass", "ground", "other", "person", "skin"):
        if name in m:
            a = 0.28 if name == "person" else 0.45
            o = o * (1 - a * m[name][..., None]) + np.array(colors[name], np.float32) * a * m[name][..., None]
    for f in _faces_at(an, scale):
        cv2.polylines(o, [cv2.convexHull(f.pts[sc.OVAL].astype(np.int32))], True, (1, 1, 0), max(2, o.shape[1] // 600))
    return adjust.to_image(o)


def mask_overlay(an: Analysis, name: str, strength: float = 100, feather: float = 0, refine: float = 0,
                 invert_mask: bool = False, edge: int = 1200, strokes=None) -> adjust.Image.Image:
    """One mask as the user's controls currently shape it: the selected area is tinted, the rest is dimmed."""
    from params import Region
    src, size, _ = _target(an, edge)
    m = region_mask(an, name, size, src, Region(mask_strength=strength, feather=feather, refine=refine, invert=invert_mask, strokes=strokes or []))
    if m is None:
        raise ValueError(f"No '{name}' mask in this photo")
    tint = np.array([1.0, 0.35, 0.1], np.float32)
    o = src * (1 - 0.55 * m[..., None]) + tint * 0.55 * m[..., None]
    o = o * (0.55 + 0.45 * m[..., None])
    return adjust.to_image(o)


def custom_overlay(an: Analysis, layer, edge: int = 1200) -> adjust.Image.Image:
    """What a custom mask selects, for the editor: the selection is tinted red and everything else is dimmed."""
    from params import MaskLayer
    src, size, _ = _target(an, edge)
    m = custom_masks.build(layer, size, src, an.bank)
    if m is None:
        raise ValueError("Unknown mask type")
    tint = np.array([1.0, 0.25, 0.15], np.float32)
    o = src * (1 - 0.6 * m[..., None]) + tint * 0.6 * m[..., None]
    o = o * (0.5 + 0.5 * m[..., None])
    return adjust.to_image(o)
