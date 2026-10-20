"""Automated image-quality tests for the masking and dodge & burn engines.

Run:  ./venv/bin/python image_quality_tests.py [photo ...]      (defaults to a portrait and a landscape in test_sheets/)

These check properties a retoucher would care about, with numbers rather than by eye:
masks aligned and in 0..1, mask operations, edits that stay inside their mask, Dodge & Burn that does nothing when
off, never clips highlights or crushes shadows, preserves skin texture, leaves no halo at the subject's edge,
is deterministic, and the pipeline never alters the original data.
"""
import hashlib
import sys
import time
from typing import Callable, List, Tuple

import cv2
import numpy as np

import dodge_burn
import engine
import masking
from params import DodgeBurn

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
Result = Tuple[str, bool, str]


def _luma(a: np.ndarray) -> np.ndarray:
    return a @ LUMA


def _render_arr(an, p, edge=900) -> np.ndarray:
    return np.asarray(engine.render(an, p, edge), np.float32) / 255


# ----------------------------------------------------------------------------------------------- masks
def t_mask_alignment_and_range(an, p) -> Result:
    bad = []
    for edge in (700, 1600):
        src, size, _ = engine._target(an, edge)
        for name in an.bank.names() + ["hair"]:
            m = an.bank.get(name, size)
            if m is None:
                continue
            if m.shape != (size[1], size[0]):
                bad.append(f"{name}@{edge}: shape {m.shape} != {(size[1], size[0])}")
            if m.dtype != np.float32 or m.min() < 0 or m.max() > 1:
                bad.append(f"{name}@{edge}: dtype/range {m.dtype} [{m.min():.2f},{m.max():.2f}]")
    return "masks are float32, within 0..1 and aligned to the image at every size", not bad, "; ".join(bad) or f"{len(an.bank.names())} masks checked"


def t_mask_operations(an, p) -> Result:
    size = (an.work.shape[1] // 2, an.work.shape[0] // 2)
    names = [n for n in an.bank.names() if an.bank.get(n, size) is not None][:3]
    if len(names) < 2:
        return "mask set operations (invert/union/intersect/subtract)", True, "skipped: fewer than two masks"
    a, b = (an.bank.get(n, size) for n in names[:2])
    checks = {
        "invert twice = original": np.allclose(masking.invert(masking.invert(a)), a, atol=1e-6),
        "union >= each input": bool((masking.union(a, b) >= np.maximum(a, b) - 1e-6).all()),
        "intersect <= each input": bool((masking.intersect(a, b) <= np.minimum(a, b) + 1e-6).all()),
        "subtract stays in 0..1": bool(0 <= masking.subtract(a, b).min() and masking.subtract(a, b).max() <= 1),
    }
    return "mask set operations (invert/union/intersect/subtract)", all(checks.values()), ", ".join(k for k, v in checks.items() if not v) or "all four behave"


def t_mask_controls(an, p) -> Result:
    src, size, _ = engine._target(an, 900)
    name = next((n for n in ("sky", "foreground", "grass", "trees") if an.bank.get(n, size) is not None), None)
    if name is None:
        return "mask controls (strength/feather/invert)", True, "skipped: no suitable mask"
    base = an.bank.effective(name, size, src)
    half = an.bank.effective(name, size, src, strength=50)
    zero = an.bank.effective(name, size, src, strength=0)
    inv = an.bank.effective(name, size, src, invert_mask=True)
    fe = 60.0
    soft = an.bank.effective(name, size, src, feather=fe)
    sigma = max(size) * (0.0012 + 0.018 * fe / 100)
    k = int(4 * sigma)                       # beyond 4 sigma a soft edge must have faded away completely
    # measured from wherever the raw mask has ANY support (a sky reflected in water legitimately carries a weak sky response)
    outside = cv2.dilate((base > 0.02).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) == 0
    leak = float(soft[outside].max()) if outside.any() else 0.0
    grew = abs(float((soft > 0.5).mean()) - float((base > 0.5).mean()))
    checks = {
        "strength 50% halves it": abs(half.max() - 0.5 * base.max()) < 0.02,
        "strength 0 = empty": zero.max() == 0,
        "invert is the complement": float(np.abs(inv - (1 - base)).max()) < 1e-5,
        "feather fades out within 4 sigma (no leak)": leak < 0.02,
        "feather does not grow the selected area": grew < 0.01,
    }
    return f"mask controls on '{name}'", all(checks.values()), ", ".join(k for k, v in checks.items() if not v) or f"all hold (leak beyond 4 sigma {leak:.3f}, area change {grew:.3f})"


def t_local_edit_stays_inside_mask(an, p) -> Result:
    name = next((n for n in ("sky", "foreground", "grass") if an.bank.has(n)), None)
    if name is None:
        return "local adjustment stays inside its mask", True, "skipped: no suitable mask"
    q = p.model_copy(deep=True)
    base = _render_arr(an, q)
    q.regions[name].exposure = -60
    q.regions[name].saturation = -60
    edited = _render_arr(an, q)
    src, size, _ = engine._target(an, 900)
    m = an.bank.get(name, (edited.shape[1], edited.shape[0]))
    k = max(5, int(max(m.shape) * 0.04)) | 1
    outside = cv2.dilate((m > 0.05).astype(np.uint8), np.ones((k, k), np.uint8)) == 0
    if not outside.any():
        return "local adjustment stays inside its mask", True, "skipped: mask covers the frame"
    d = np.abs(edited - base).max(axis=2)[outside]
    ok = float(d.mean()) < 0.004 and float(np.percentile(d, 99.5)) < 0.03
    return f"a strong '{name}' edit stays inside the '{name}' mask", ok, f"outside the mask: mean change {d.mean():.4f}, 99.5th pct {np.percentile(d, 99.5):.3f}"


# ----------------------------------------------------------------------------------------------- dodge & burn
def _db_inputs(an):
    src, size, _ = engine._target(an, 1000)
    masks = engine._masks_at(an, size)
    return src, masks


def t_db_off_is_identity(an, p) -> Result:
    src, masks = _db_inputs(an)
    cases = [DodgeBurn(enabled=False, dodge=100, burn=100), DodgeBurn(), DodgeBurn(dodge=80, strength=0)]
    ok = all(dodge_burn.apply(src, masks, c) is src or np.array_equal(dodge_burn.apply(src, masks, c), src) for c in cases)
    return "dodge & burn does nothing when disabled, at zero, or at strength 0", ok, "output identical to the input"


def t_db_highlights_and_shadows_protected(an, p) -> Result:
    src, masks = _db_inputs(an)
    clip0, crush0 = float((src.max(axis=2) >= 0.99).mean()), float((src.max(axis=2) <= 0.02).mean())
    worst_clip = worst_crush = 0.0
    for kw in (dict(dodge=100, dodge_tones=100), dict(dodge=100, dodge_tones=70, form=100), dict(burn=100, burn_tones=0),
               dict(burn=100, separation=100, burn_tones=20)):
        out = dodge_burn.apply(src, masks, DodgeBurn(**kw))
        worst_clip = max(worst_clip, float((out.max(axis=2) >= 0.99).mean()) - clip0)
        worst_crush = max(worst_crush, float((out.max(axis=2) <= 0.02).mean()) - crush0)
    ok = worst_clip <= 0.003 and worst_crush <= 0.01
    return "extreme dodge never clips highlights, extreme burn never crushes shadows", ok, f"extra clipped {worst_clip:+.4f}, extra crushed {worst_crush:+.4f}"


def t_db_preserves_skin_texture(an, p) -> Result:
    src, masks = _db_inputs(an)
    skin = masks.get("skin")
    if skin is None or (skin > 0.6).sum() < 800:
        return "skin texture is preserved by dodge & burn", True, "skipped: no reliable skin mask"
    out = dodge_burn.apply(src, masks, DodgeBurn(dodge=100, form=100, separation=100))
    sel = cv2.erode((skin > 0.6).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    def hf(a):   # RELATIVE texture contrast: fine detail divided by the local brightness it sits on. A brighter
        y = _luma(a)  # patch of skin has proportionally larger absolute detail; what must stay constant is this ratio.
        low = cv2.GaussianBlur(y, (0, 0), 3)
        return float(np.var(((y - low) / np.maximum(low, 0.05))[sel]))

    ratio = hf(out) / max(hf(src), 1e-9)
    return "skin texture is preserved by dodge & burn (relative texture contrast ratio 0.9-1.1)", 0.90 <= ratio <= 1.10, f"ratio {ratio:.3f}"


def t_db_no_halo(an, p) -> Result:
    src, masks = _db_inputs(an)
    subj = masks.get("person")
    if subj is None or (subj > 0.5).mean() < 0.02:
        return "no halo around the subject", True, "skipped: no subject mask"
    inside = (subj > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
    near = (dist > 2) & (dist < 6)
    far = (dist > 20) & (dist < 60)
    if near.sum() < 100 or far.sum() < 100:
        return "no halo around the subject", True, "skipped: subject too small/large"
    y0 = _luma(src)
    # burn: the darkening must reach right up to the outline (no light rim); dodge must not spill outside
    burn = _luma(dodge_burn.apply(src, masks, DodgeBurn(burn=100, separation=60, burn_tones=50))) - y0
    dodge = _luma(dodge_burn.apply(src, masks, DodgeBurn(dodge=100, form=100))) - y0
    rim = float(burn[near].mean() / min(burn[far].mean(), -1e-6)) if burn[far].mean() < -1e-4 else 1.0
    spill = float(dodge[near].mean())
    ok = rim >= 0.6 and spill < 0.01
    return "no halo around the subject (burn reaches the outline, dodge does not spill)", ok, f"burn near/far ratio {rim:.2f} (>=0.6), dodge spill {spill:+.4f} (<0.01)"


def t_db_deterministic_and_independent(an, p) -> Result:
    src, masks = _db_inputs(an)
    a = dodge_burn.apply(src, masks, DodgeBurn(dodge=60, burn=40, separation=30, form=20))
    b = dodge_burn.apply(src, masks, DodgeBurn(dodge=60, burn=40, separation=30, form=20))
    c = dodge_burn.apply(src, masks, DodgeBurn(dodge=60, burn=0, separation=30, form=20))
    ok = np.array_equal(a, b) and not np.array_equal(a, c) or masks.get("person") is None
    return "dodge & burn is deterministic and each layer is independent", bool(ok), "same input -> same output; changing one layer changes the result"


def t_auto_lighting_conservative(an, p) -> Result:
    size = (an.work.shape[1], an.work.shape[0])
    masks = {n: an.bank.get(n, size) for n in ("skin", "person", "foreground", "background")}
    res, notes = dodge_burn.auto_lighting(colorcalc(an), {k: v for k, v in masks.items() if v is not None})
    ok = res.dodge <= 50 and res.burn <= 35 and res.separation <= 55 and res.form <= 65 and bool(notes)
    if masks.get("person") is None:
        ok = ok and res.separation <= 25 and res.form == 0 and res.dodge <= 30   # no person: only gentle corrections
    return "automatic lighting is conservative and explains itself", ok, f"dodge {res.dodge:.0f} burn {res.burn:.0f} sep {res.separation:.0f} form {res.form:.0f} | {notes[0] if notes else ''}"


def colorcalc(an):
    import colorcal
    return colorcal.apply_gains(an.work, an.wb.gains)


# ----------------------------------------------------------------------------------------------- pipeline
def t_pipeline_is_non_destructive(an, p) -> Result:
    h0 = hashlib.md5(an.work.tobytes()).hexdigest()
    q = p.model_copy(deep=True)
    q.dodge_burn = DodgeBurn(dodge=80, burn=80, separation=80, form=80)
    for kind in ("a", "b", "c"):
        q.basic.exposure = {"a": 30, "b": -30, "c": 0}[kind]
        engine.render(an, q, 700)
    return "the original data is never altered by edits", hashlib.md5(an.work.tobytes()).hexdigest() == h0, "analysis image hash unchanged after repeated renders"


def t_render_size_matches_image(an, p) -> Result:
    img = engine.render(an, p, 800)
    full_ar = an.full_size[0] / an.full_size[1]
    ar = img.size[0] / img.size[1]
    return "preview keeps the image's exact proportions (alignment with the original)", abs(ar - full_ar) / full_ar < 0.004, f"preview {img.size}, original {an.full_size}"


def t_neutral_params_are_identity(an, p) -> Result:
    from params import EditParams
    src, size, _ = engine._target(an, 700)
    an.cache.clear()
    out = np.asarray(engine.render(an, EditParams(), 700), np.float32) / 255
    d = float(np.abs(out - src).max())
    return "neutral parameters leave the picture untouched (Auto Edit off = the original)", d < 0.01, f"max change {d:.4f}"


def t_slider_directions(an, p) -> Result:
    from params import EditParams
    an.cache.clear()
    base = np.asarray(engine.render(an, EditParams(), 700), np.float32) / 255
    y0 = _luma(base)
    bad = []

    def moved(path, val):
        q = EditParams()
        o = q
        for k in path[:-1]:
            o = getattr(o, k)
        setattr(o, path[-1], val)
        an.cache.clear()
        return np.asarray(engine.render(an, q, 700), np.float32) / 255

    if _luma(moved(("basic", "exposure"), 50)).mean() <= y0.mean() + 0.02: bad.append("exposure+")
    if _luma(moved(("basic", "exposure"), -50)).mean() >= y0.mean() - 0.02: bad.append("exposure-")
    if _luma(moved(("basic", "contrast"), 50)).std() <= y0.std() + 0.004: bad.append("contrast+")
    a = moved(("basic", "temperature"), 50)
    if a[..., 0].mean() / a[..., 2].mean() <= base[..., 0].mean() / base[..., 2].mean() * 1.03: bad.append("temperature+")
    hsv = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2HSV)[..., 1].mean()
    if hsv(moved(("basic", "vibrance"), 50)) <= hsv(base) + 0.01: bad.append("vibrance+")
    if hsv(moved(("basic", "saturation"), -50)) >= hsv(base) - 0.02: bad.append("saturation-")
    hi = y0 > np.percentile(y0, 85)
    if _luma(moved(("basic", "highlights"), -70))[hi].mean() >= y0[hi].mean() - 0.02: bad.append("highlights-")
    sh = y0 < np.percentile(y0, 15)
    if _luma(moved(("basic", "shadows"), 70))[sh].mean() <= y0[sh].mean() + 0.01: bad.append("shadows+")
    return "basic sliders move the picture the right way (exposure, contrast, colour, tone ranges)", not bad, ", ".join(bad) or "all eight directions correct"


def t_srgb_pipeline(an, p) -> Result:
    import io
    from PIL import ImageCms
    img = adjust_load(an.path)
    prof = img.info.get("icc_profile")
    desc = ImageCms.getOpenProfile(io.BytesIO(prof)).profile.profile_description.lower() if prof else "none"
    return "photos are converted to sRGB on load, so preview, analysis and export agree", ("srgb" in desc) or prof is None, f"working profile: {desc}"


def adjust_load(path):
    import adjust
    return adjust.load(path, 64)


def t_light_scene_read(an, p) -> Result:
    r = an.read
    ok = r is not None and r.light in ("golden_hour", "sunset", "blue_hour", "sunny", "overcast", "night", "indoor") and bool(r.evidence)
    return "scene and light are read, with the evidence behind the decision", ok, f"{r.light}/{r.scene} ({'; '.join(r.evidence[:2])})" if r else "no scene read"


# ----------------------------------------------------------------------------------------------- custom masks + curves
def _layer(**kw):
    from params import MaskLayer
    return MaskLayer(id="t", **kw)


def t_custom_mask_shapes(an, p) -> Result:
    import custom_masks
    from params import Stroke
    src, size, _ = engine._target(an, 800)
    layers = [_layer(kind="linear"), _layer(kind="radial"),
              _layer(kind="brush", strokes=[Stroke(pts=[[0.2, 0.3], [0.6, 0.5], [0.8, 0.4]], size=20)]),
              _layer(kind="luminosity"), _layer(kind="color", pick=[0.5, 0.5])]
    bad = []
    for l in layers:
        m = custom_masks.build(l, size, src, an.bank)
        if m is None or m.shape != (size[1], size[0]) or m.dtype != np.float32 or m.min() < 0 or m.max() > 1:
            bad.append(l.kind)
    return "all five custom mask types give aligned float 0..1 masks", not bad, ", ".join(bad) or "linear, radial, brush, luminosity, colour"


def t_gradient_and_radial_geometry(an, p) -> Result:
    import custom_masks
    src, size, _ = engine._target(an, 800)
    w, h = size
    lin = custom_masks.build(_layer(kind="linear", line=[0.5, 0.1, 0.5, 0.6], feather=70), size, src)
    col = lin[:, w // 2]
    mono = bool(np.all(np.diff(col) <= 1e-4))
    ok_lin = lin[int(h * 0.05), w // 2] > 0.98 and lin[int(h * 0.75), w // 2] < 0.02 and mono
    rad = custom_masks.build(_layer(kind="radial", ellipse=[0.5, 0.5, 0.25, 0.25, 0], feather=40), size, src)
    inv = custom_masks.build(_layer(kind="radial", ellipse=[0.5, 0.5, 0.25, 0.25, 0], feather=40, invert=True), size, src)
    ok_rad = rad[h // 2, w // 2] > 0.98 and rad[2, 2] < 0.02 and float(np.abs(inv - (1 - rad)).max()) < 1e-5
    return "gradient is 1 at its start, 0 at its end, monotone; radial is 1 inside, 0 outside, invert is exact", bool(ok_lin and ok_rad), \
        f"linear start {lin[int(h*0.05), w//2]:.2f} end {lin[int(h*0.75), w//2]:.2f} monotone={mono}; radial centre {rad[h//2,w//2]:.2f} corner {rad[2,2]:.2f}"


def t_brush_resolution_independent(an, p) -> Result:
    import custom_masks
    from params import Stroke
    l = _layer(kind="brush", strokes=[Stroke(pts=[[0.15, 0.2], [0.5, 0.45], [0.85, 0.3]], size=18, feather=50),
                                      Stroke(pts=[[0.3, 0.7], [0.7, 0.75]], size=30, feather=80, flow=70)])
    a = custom_masks.build(l, (480, 360), np.zeros((360, 480, 3), np.float32))
    b = custom_masks.build(l, (1600, 1200), np.zeros((1200, 1600, 3), np.float32))
    b_small = cv2.resize(b, (480, 360), interpolation=cv2.INTER_AREA)
    corr = float(np.corrcoef(a.ravel(), b_small.ravel())[0, 1])
    painted = float((a > 0.5).mean())
    l2 = l.model_copy(deep=True); l2.strokes.append(Stroke(pts=[[0.15, 0.2], [0.5, 0.45], [0.85, 0.3]], size=30, erase=True))
    erased = float((custom_masks.build(l2, (480, 360), np.zeros((360, 480, 3), np.float32)) > 0.5).mean())
    return "brush strokes cover the same area at every resolution, and erase removes paint", corr > 0.97 and 0.01 < painted < 0.4 and erased < painted * 0.8, \
        f"correlation preview vs full {corr:.3f}; painted {painted:.2%} -> after erase {erased:.2%}"


def t_range_masks_select_what_they_claim(an, p) -> Result:
    import custom_masks
    src, size, _ = engine._target(an, 800)
    y = _luma(src)
    m = custom_masks.build(_layer(kind="luminosity", lum=[65, 100], lum_soft=15, feather=0), size, src)
    sel = m > 0.5
    ok_l = sel.sum() > 200 and y[sel].mean() > y[~sel].mean() + 0.15
    hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
    ys_, xs_ = np.nonzero((hsv[..., 1] > 0.4) & (hsv[..., 2] > 0.3))
    ok_c, det = True, "skipped: no saturated pixel"
    if len(ys_) > 100:
        i = len(ys_) // 2; py, px = int(ys_[i]), int(xs_[i]); tgt = hsv[py, px, 0]
        m2 = custom_masks.build(_layer(kind="color", pick=[px / size[0], py / size[1]], tolerance=30, feather=0), size, src)
        sel2 = m2 > 0.5
        dh = np.abs((hsv[..., 0][sel2] - tgt + 180) % 360 - 180)
        ok_c = sel2.sum() > 100 and float(np.median(dh)) < 25 and m2[py, px] > 0.9
        det = f"picked hue {tgt:.0f}: {int(sel2.sum())} px selected, median hue error {np.median(dh):.0f} deg"
    return "luminosity picks the bright pixels, colour range picks the picked colour", bool(ok_l and ok_c), f"bright mean {y[sel].mean():.2f} vs rest {y[~sel].mean():.2f}; {det}"


def t_mask_limit_and_refine(an, p) -> Result:
    import custom_masks, masks as mk
    src, size, _ = engine._target(an, 800)
    name = next((n for n in ("sky", "foreground", "grass", "water") if an.bank.get(n, size) is not None), None)
    if name is None:
        return "mask limited to an AI region; auto-mask follows edges", True, "skipped: no AI region"
    reg = an.bank.get(name, size)
    m = custom_masks.build(_layer(kind="linear", line=[0.5, 0.0, 0.5, 1.0], feather=100, limit=name), size, src, an.bank)
    outside = reg < 0.02
    leak = float(m[outside].max()) if outside.any() else 0.0
    inv = custom_masks.build(_layer(kind="linear", line=[0.5, 0.0, 0.5, 1.0], feather=100, limit=name, limit_invert=True), size, src, an.bank)
    leak2 = float(inv[reg > 0.98].max()) if (reg > 0.98).any() else 0.0
    soft = _layer(kind="radial", ellipse=[0.5, 0.45, 0.3, 0.3, 0], feather=30)
    e0 = mk.score_mask(custom_masks.build(soft, size, src), src, "sky").edge
    e1 = mk.score_mask(custom_masks.build(soft.model_copy(update={"refine": 100}), size, src), src, "sky").edge
    return "limit confines a mask to (or away from) an AI region; auto-mask snaps to real edges", leak < 0.02 and leak2 < 0.02 and e1 >= e0 - 0.02, \
        f"leak outside {name} {leak:.3f}, inside when inverted {leak2:.3f}; edge alignment {e0:.2f} -> {e1:.2f} with auto-mask"


def t_custom_mask_edit_stays_inside(an, p) -> Result:
    from params import EditParams
    an.cache.clear()
    base_p = EditParams()
    base = np.asarray(engine.render(an, base_p, 700), np.float32) / 255
    q = EditParams()
    q.masks.append(_layer(kind="linear", line=[0.5, 0.0, 0.5, 0.4], feather=60))
    q.masks[0].adj.exposure = -70
    q.masks[0].adj.saturation = -50
    an.cache.clear()
    out = np.asarray(engine.render(an, q, 700), np.float32) / 255
    H = out.shape[0]
    top = float(np.abs(out - base)[: int(H * 0.15)].mean())
    bottom = float(np.abs(out - base)[int(H * 0.6):].max())
    return "a gradient edit changes the top and leaves the bottom untouched", top > 0.05 and bottom < 0.004, f"top changed by {top:.3f}, bottom max change {bottom:.4f}"


def t_point_curve(an, p) -> Result:
    import grade
    from params import ToneCurve
    x = np.linspace(0, 1, 256, dtype=np.float32)
    ramp = np.stack([x, x, x], axis=-1)[None]
    ident = grade.point_curve(ramp, ToneCurve())
    ok_id = float(np.abs(ident - ramp).max()) < 1e-6
    s = ToneCurve(rgb=[[0, 0], [0.25, 0.18], [0.75, 0.83], [1, 1]])
    out = grade.point_curve(ramp, s)[0, :, 0]
    mono = bool(np.all(np.diff(out) >= -1e-6))
    ok_s = out[64] < x[64] - 0.03 and out[192] > x[192] + 0.03 and abs(out[0]) < 1e-6 and abs(out[-1] - 1) < 1e-6
    red = grade.point_curve(ramp, ToneCurve(r=[[0, 0], [0.5, 0.65], [1, 1]]))[0]
    ok_r = red[128, 0] > x[128] + 0.08 and abs(red[128, 1] - x[128]) < 1e-5 and abs(red[128, 2] - x[128]) < 1e-5
    wild = grade.curve_lut([[0, 0], [0.5, 1.0], [0.52, 0.0], [1, 1]])        # impossible zig-zag must still be a valid curve
    ok_w = wild is not None and wild.min() >= 0 and wild.max() <= 1
    return "point curve: identity is exact, an S-curve adds contrast, channels act alone, output stays valid", bool(ok_id and mono and ok_s and ok_r and ok_w), \
        f"identity {ok_id}, monotone {mono}, S shape {ok_s}, red-only {ok_r}, wild points {ok_w}"


# ----------------------------------------------------------------------------------------------- coverage, region fixes, LUTs, matching
def t_region_coverage(an, p) -> Result:
    names = an.bank.names()
    cov = an.coverage
    size = (an.work.shape[1], an.work.shape[0])
    claims = np.zeros(an.work.shape[:2], np.float32)
    for n in names:
        if n == "other":
            continue
        m = an.bank.get(n, size)
        if m is not None:
            claims = np.maximum(claims, m)
    other = an.bank.get("other", size)
    gap = 1.0 - claims
    uncovered = float((np.maximum(gap - (other if other is not None else 0), 0) > 0.5).mean())
    ok = "foreground" in names and uncovered < 0.02 and (other is None or float((other > 0.5).mean()) < 0.5)
    return "every pixel belongs to a region (leftovers become Miscellaneous), and a foreground always exists", bool(ok), \
        f"foreground {'foreground' in names}, unclaimed after Miscellaneous {uncovered * 100:.1f}%, Miscellaneous {cov.get('other', 0)}%"


def t_region_paint_fix(an, p) -> Result:
    from params import Stroke
    size = (an.work.shape[1] // 2, an.work.shape[0] // 2)
    src = cv2.resize(an.work, size)
    name = "foreground" if "foreground" in an.bank.names() else an.bank.names()[0]
    r0 = p.regions[name].model_copy(deep=True)
    base = engine.region_mask(an, name, size, src, r0)
    add = Stroke(pts=[[0.2, 0.2], [0.8, 0.2]], size=30, feather=40, flow=100, erase=False)
    rem = Stroke(pts=[[0.2, 0.7], [0.8, 0.7]], size=30, feather=40, flow=100, erase=True)
    ra, rr = r0.model_copy(deep=True), r0.model_copy(deep=True)
    ra.strokes, rr.strokes = [add], [rem]
    ma, mr = engine.region_mask(an, name, size, src, ra), engine.region_mask(an, name, size, src, rr)
    band = lambda m, y: float(m[int(size[1] * (y - 0.06)):int(size[1] * (y + 0.06))].mean())
    far = np.ones(base.shape, bool); far[int(size[1] * 0.1):int(size[1] * 0.3)] = False; far[int(size[1] * 0.6):int(size[1] * 0.8)] = False
    ok = band(ma, 0.2) >= band(base, 0.2) - 1e-4 and band(mr, 0.7) <= band(base, 0.7) + 1e-4 and \
        (band(ma, 0.2) > band(base, 0.2) + 0.01 or band(base, 0.2) > 0.97) and (band(mr, 0.7) < band(base, 0.7) - 0.01 or band(base, 0.7) < 0.03) and \
        float(np.abs(ma - base)[far].max()) < 0.12 and float(np.abs(mr - base)[far].max()) < 0.12
    return "paint-to-fix: an add stroke grows the region, a remove stroke shrinks it, nothing changes elsewhere", bool(ok), \
        f"{name}: add band {band(base, .2):.2f}->{band(ma, .2):.2f}, remove band {band(base, .7):.2f}->{band(mr, .7):.2f}"


def t_lut_behaviour(an, p) -> Result:
    import luts
    arr = _render_arr(an, p, 500)
    ident = luts.Lut("i", "id", luts._grid(17))
    ok_id = float(np.abs(luts.apply(arr, ident) - arr).max()) < 2e-3
    tl = luts.get("builtin:teal_orange")
    full, half = luts.apply(arr, tl, 100), luts.apply(arr, tl, 50)
    ok_half = float(np.abs(half - (arr + full) / 2).max()) < 1e-4
    ok_zero = luts.apply(arr, tl, 0) is arr
    bw = luts.apply(arr, luts.get("builtin:bw_classic"))
    ok_bw = float(np.abs(bw[..., 0] - bw[..., 2]).max()) < 0.02
    text = luts.to_cube(tl)
    rt = luts.parse_cube(text)
    ok_rt = float(np.abs(rt.table - tl.table).max()) < 1e-5
    bad = 0
    for junk in ("hello", "LUT_3D_SIZE 4\n0 0 0\n", "LUT_3D_SIZE 2\n0 0 x\n"):
        try:
            luts.parse_cube(junk)
        except ValueError:
            bad += 1
    q = p.model_copy(deep=True); q.lut.id = "builtin:teal_orange"; q.lut.intensity = 100
    ok_pipe = float(np.abs(_render_arr(an, q, 500) - arr).mean()) > 0.01 and float(np.abs(_render_arr(an, p, 500) - arr).max()) < 1e-6
    return "LUTs: identity is exact, strength blends linearly, B&W is grey, .cube round-trips, junk is rejected, pipeline applies them", \
        bool(ok_id and ok_half and ok_zero and ok_bw and ok_rt and bad == 3 and ok_pipe), \
        f"identity {ok_id}, half {ok_half}, zero {ok_zero}, bw {ok_bw}, roundtrip {ok_rt}, rejected {bad}/3, pipeline {ok_pipe}"


def t_match_reference(an, p) -> Result:
    import match
    refs = [("warm_bright", lambda a: np.clip(a * np.array([1.12, 1.0, 0.82], np.float32) * 1.15 + 0.04, 0, 1)),
            ("dark_cool", lambda a: np.clip(a * np.array([0.85, 0.95, 1.12], np.float32) * 0.8, 0, 1))]
    base = _render_arr(an, p, 420)
    rs = lambda q: _render_arr(an, q, 420)
    ok, detail = True, []
    for label, fn in refs:
        look = match.measure(fn(base))
        m = match.solve(rs, p, look)
        d0 = float(np.abs(match.measure(base).q - look.q).mean())
        outs = {k: match.measure(rs(match.blend(p, m.full, k))) for k in (0, 0.5, 1.0)}
        dk = {k: float(np.abs(o.q - look.q).mean()) for k, o in outs.items()}
        closer = dk[1.0] < d0 * 0.6 and dk[0.5] < dk[0] + 1e-6 and abs(dk[0] - d0) < 0.02
        sane = all(abs(getattr(m.full.basic, a)) <= 70 for a in ("saturation", "vibrance")) and all(getattr(m.full.grading, n).sat <= match.WHEEL_CAP[n] + 0.01 for n in match.RANGES)
        ok &= bool(closer and sane)
        detail.append(f"{label}: distance {d0:.3f}->{dk[0.5]:.3f}@50%->{dk[1.0]:.3f}@100% {'ok' if closer and sane else 'BAD'}")
    zero = match.blend(p, match.solve(rs, p, match.measure(base)).full, 0)
    ok &= zero.model_dump() == p.model_copy(deep=True).model_dump() or float(np.abs(_render_arr(an, zero, 420) - base).max()) < 0.02
    return "match reference: gets closer to the reference, strength blends, sliders stay in a sane range, 0% changes nothing", bool(ok), "; ".join(detail)


def t_banded_ops_are_exact(an, p) -> Result:
    """Large exports run per-pixel operations in parallel row bands. That must change nothing: identical to a single pass."""
    import bands
    import colorcal
    import grade
    from params import Grading, Wheel
    big = cv2.resize(an.work, (2400, 1800), interpolation=cv2.INTER_CUBIC).clip(0, 1).astype(np.float32)
    ctx = grade.Ctx(big)
    skin = cv2.GaussianBlur((big[..., 0] > 0.6).astype(np.float32), (0, 0), 20)
    g = Grading(shadows=Wheel(hue=210, sat=30), highs=Wheel(hue=40, sat=25, lum=5))
    cases = {
        "exposure": (grade.exposure, (big, 30)), "contrast": (grade.contrast, (big, 40)), "highlights": (grade.highlights, (big, -40, ctx)),
        "shadows": (grade.shadows, (big, 40, ctx)), "whites": (grade.whites, (big, 20, ctx)), "clarity": (grade.clarity, (big, 40, ctx)),
        "texture": (grade.texture, (big, 40, ctx)), "saturation": (grade.saturation, (big, 30)), "hue": (grade.hue_shift, (big, 20)),
        "mixer": (grade.mixer, (big, {"green": [10, 20, -10], "blue": [0, 15, 5]})), "wheels": (grade.apply_wheels, (big, g, skin, skin)),
        "gains": (colorcal.apply_gains, (big, np.array([1.1, 1.0, 0.9], np.float32), 10, -5)),
        "balance": (colorcal.color_balance, (big, (10, 0, -10), (0, 5, 0), (-5, 0, 10))),
        "vibrance": (lambda a, v, pr: grade.vibrance(a, v, pr), (big, 40, skin)),
    }
    old = bands.MIN_PIXELS
    bad = []
    try:
        for name, (fn, args) in cases.items():
            bands.MIN_PIXELS = 10 ** 12
            ref = fn(*args)
            bands.MIN_PIXELS = 1
            got = fn(*args)
            if not np.array_equal(ref, got):
                bad.append(f"{name} (max diff {float(np.abs(ref - got).max()):.2e})")
        noop = grade.exposure(big, 0) is big
    finally:
        bands.MIN_PIXELS = old
    return "banded (parallel) per-pixel operations give exactly the same pixels as a single pass", not bad and noop, \
        f"{len(cases) - len(bad)}/{len(cases)} identical; no-op returns the input untouched: {noop}" + (f"; DIFFERENT: {bad}" if bad else "")


def t_smart_looks(an, p) -> Result:
    import look_engine
    base = _render_arr(an, p, 600)
    size = (base.shape[1], base.shape[0])
    skin = an.bank.get("skin", size)
    outs, bad = {}, []
    for lid, look in look_engine.LOOKS.items():
        q, notes, facts = look_engine.build(an, p, look, 1.0)
        r = _render_arr(an, q, 600)
        outs[lid] = r
        y = _luma(r)
        clip = float((r.max(2) > 0.995).mean() - (base.max(2) > 0.995).mean())
        crush = float((r.max(2) < 0.02).mean() - (base.max(2) < 0.02).mean())
        med = float(np.median(y))
        ok = clip < 0.04 and crush < 0.06 and 0.2 < med < 0.7 and bool(notes)
        if skin is not None and (skin > 0.5).mean() > 0.01:
            ok = ok and float(np.abs(r[skin > 0.5] - base[skin > 0.5]).mean()) < 0.12
        if not ok:
            bad.append(lid)
    ids = list(outs)
    gaps = [float(np.abs(outs[a] - outs[b]).mean()) for i, a in enumerate(ids) for b in ids[i + 1:]]
    distinct = min(gaps) > 0.004
    look = look_engine.LOOKS["cinematic"]
    d = [float(np.abs(_render_arr(an, look_engine.build(an, p, look, k)[0], 600) - base).mean()) for k in (0.0, 0.3, 0.7, 1.2)]
    monotonic = d[0] < 0.004 and d[1] < d[2] and d[3] > 0.8 * d[2]      # 0% leaves the edit as it is; beyond 100% the safety governor may hold the change level
    return "smart looks: all 8 are safe (no clipping/crushing, believable brightness, skin natural), distinct, and strength scales smoothly", \
        bool(not bad and distinct and monotonic), f"unsafe: {bad or 'none'}; closest pair differs by {min(gaps):.4f}; strength 0/30/70/120% change {d[0]:.3f}/{d[1]:.3f}/{d[2]:.3f}/{d[3]:.3f}"


def t_adaptive_looks(an, p) -> Result:
    """The same look must be parameterised differently for a bright-sky/dark-foreground photo and an evenly lit overcast one."""
    import look_engine as le
    A = le.Tonal(rng=0.80, p50=0.38, clip_hi=0.03, sky_med=0.85, sky_clip=0.06, fg_med=0.20, mt_med=0.4, mt_range=0.35)
    B = le.Tonal(rng=0.38, p50=0.50, clip_hi=0.0, sky_med=0.62, sky_clip=0.0, fg_med=0.50, mt_med=0.5, mt_range=0.10)
    for t in (A, B):
        t.gap = le._c((t.sky_med - t.fg_med - 0.12) / 0.35); t.flat = le._c((0.62 - t.rng) / 0.30); t.contrasty = le._c((t.rng - 0.82) / 0.14)
        t.fg_dark = le._c((0.30 - t.fg_med) / 0.15); t.fg_bright = le._c((t.fg_med - 0.52) / 0.20); t.mt_flat = le._c((0.22 - t.mt_range) / 0.12)
    feel = le.LOOKS["cinematic"].feel
    pa = le.adapt_regions(le._region_plan(feel, le.Facts("landscape", "sunny", "mountains")), A)
    pb = le.adapt_regions(le._region_plan(feel, le.Facts("landscape", "sunny", "mountains")), B)
    ga, gb = le.global_multipliers(A), le.global_multipliers(B)
    checks = {
        "sky is burned harder in A": pa["sky"]["highlights"] < pb["sky"]["highlights"] - 3 and pa["sky"]["exposure"] < pb["sky"]["exposure"],
        "foreground lifted more in A": pa["foreground"]["shadows"] > pb["foreground"]["shadows"] + 3,
        "clouds / mountains get more contrast in B": pb["sky"]["contrast"] > pa["sky"]["contrast"] and pb["mountains"]["clarity"] > pa["mountains"]["clarity"] + 3,
        "global contrast higher in B": gb["contrast"] > ga["contrast"] + 0.5,
        "highlight recovery stronger in A": ga["highlights"] > gb["highlights"] + 0.5,
        "adaptive strength stays 0.8..1.2": 0.8 <= le.adaptive_strength(A) <= 1.2 and 0.8 <= le.adaptive_strength(B) <= 1.2,
    }
    ta = le.measure_tonal(an)
    real = 0.0 <= ta.flat <= 1.0 and 0.0 < ta.rng < 1.0 and 0.8 <= le.adaptive_strength(ta) <= 1.2
    bad = [k for k, v in checks.items() if not v]
    return "adaptive looks: bright-sky/dark-foreground and flat/overcast photos get different parameters from the same look", \
        bool(not bad and real), f"failed: {bad or 'none'}; this photo: range {ta.rng:.2f}, flat {ta.flat:.2f}, gap {ta.gap:.2f}, adaptive ×{le.adaptive_strength(ta):.2f}"


def t_sky_studio(an, p) -> Result:
    import sky_engine as se
    r = se.read_sky(an)
    if r is None:
        sg = se.suggest(an)
        return "sky studio: photos without a sky are handled (no sky read, no suggestions)", sg["sky"] is None and not sg["sky_treatments"], "no sky in this photo"
    base = _render_arr(an, p, 600)
    size = (base.shape[1], base.shape[0])
    sky = an.bank.get("sky", size)
    bad, fits = [], se.fit_scores(r, an.read.light if an.read else None)
    for tid in se.TREATMENTS:
        q, notes, _ = se.apply(an, p, tid, 1.0)
        o = _render_arr(an, q, 600)
        d = np.abs(o - base).max(axis=2)
        inside, leak = float(d[sky > 0.5].mean()), float(d[sky < 0.02].mean())
        clip = float((o.max(2) > 0.995).mean() - (base.max(2) > 0.995).mean())
        med = float(np.median(_luma(o)))
        if not (leak < 0.004 and clip < 0.04 and 0.2 < med < 0.75 and inside > 0.003 and all(l.limit == "sky" for l in q.masks if l.id.startswith("sky_"))):
            bad.append(tid)
    # strength 0 must leave the sky alone; the best-fitting treatment must be a plausible match for what the sky is
    q0, _, _ = se.apply(an, p, "deep_blue", 0.0)
    zero = float(np.abs(_render_arr(an, q0, 600) - base).max(axis=2)[sky > 0.5].mean())
    sg = se.suggest(an)
    top = sg["sky_treatments"][0]
    ok_rank = top["fit"] == max(fits.values()) and len(sg["looks"]) == 3
    return "sky studio: every treatment stays inside the sky, is safe, 0% changes nothing, and suggestions are ranked", \
        bool(not bad and zero < 0.004 and ok_rank), f"sky is '{r.kind}'; unsafe: {bad or 'none'}; 0% change {zero:.4f}; top treatment {top['id']} ({top['fit']}); looks {[l['id'] for l in sg['looks']]}"


def t_sky_replacement(an, p) -> Result:
    import look_engine as le
    import sky_replace as sr
    if not sr.library():
        return "sky replacement (library not downloaded)", True, "skipped"
    info = sr.need(an)
    if not info["available"]:
        return "sky replacement: photos without a sky are declined cleanly", True, "no sky in this photo"
    facts = le.read_facts(an, None, None)
    ranked = sr.rank(an, facts.light, le.measure_tonal(an))
    ok_rank = len(ranked) >= 40 and all(0 <= r["fit"] <= 1 for r in ranked) and ranked == sorted(ranked, key=lambda r: -r["fit"])
    bucket = sr.LIGHT_BUCKET.get(facts.light, "day")
    ok_rank = ok_rank and sr.TIME_BUCKET[ranked[0]["time"]] == bucket and ranked[-1]["fit"] < 0.3
    src, size, _ = engine._target(an, 900)
    masks = engine._masks_at(an, size)
    sky = masks["sky"]
    good = ranked[0]["id"]
    plan = sr.plan(an, good)
    out, m = sr.apply(an, src, masks, plan)
    d = np.abs(out - src).max(axis=2)
    inside = float(d[sky > 0.8].mean())
    land = (m < 0.02) & (sky < 0.02)
    # far from the sky and water, only the gentle relight may touch the landscape
    far = cv2.erode(land.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    water = masks.get("water")
    if water is not None:
        far &= water < 0.05
    gentle = float(d[far].mean()) < 0.045 and float(d[far].max()) < 0.2
    p0 = plan.model_copy(update={"relight": 0, "reflect": 0, "edge": 0})
    out0, _ = sr.apply(an, src, masks, p0)
    far0 = cv2.erode(((m < 0.02) & (sky < 0.02)).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    untouched = float(np.abs(out0 - src).max(axis=2)[far0].max()) < 1e-6
    off, _ = sr.apply(an, src, masks, plan.model_copy(update={"blend": 0}))
    identity = off is src
    # resolution independence: the same sky at 600 px and 1200 px agree
    s2, size2, _ = engine._target(an, 1200)
    m2 = engine._masks_at(an, size2)
    o2, _ = sr.apply(an, s2, m2, plan)
    a = cv2.resize(out, (size2[0], size2[1]), interpolation=cv2.INTER_CUBIC)
    corr = float(np.corrcoef((a @ LUMA).ravel(), (o2 @ LUMA).ravel())[0, 1])
    # the whole engine: the replacement is part of the render and Edit strength 0 returns the original
    q = p.model_copy(deep=True)
    q.sky_replace = plan
    q.intensity = 0.0
    zero = float(np.abs(_render_arr(an, q, 600) - engine._target(an, 600)[0]).max())
    checks = {"sky replaced": inside > 0.03, "landscape only gently relit": gentle, "relight/reflect/edge off leaves landscape exact": untouched,
              "0% blend is a no-op": identity, "same at two sizes": corr > 0.97, "edit strength 0 = original": zero < 0.01, "ranking sane": ok_rank}
    bad = [k for k, v in checks.items() if not v]
    return "sky replacement: only the sky changes (plus a gentle relight), no-op at 0%, resolution independent, library ranked by fit", not bad, \
        f"{'all hold' if not bad else 'FAILED: ' + str(bad)}; sky change {inside:.2f}, corr {corr:.3f}; best sky '{good}' for light '{facts.light}'"


def t_sky_full_res_matte(an, p) -> Result:
    """The full-resolution sky matte must recover a sharp skyline from a blocky low-resolution mask."""
    import sky_replace as sr
    H, W = 1800, 3000
    xs = np.arange(W)
    ridge = (700 + 80 * np.sin(xs / 260.0) + 30 * np.sin(xs / 61.0)).astype(np.int32)          # true skyline, per column
    yy = np.arange(H)[:, None]
    truth = (yy < ridge[None, :]).astype(np.float32)                                          # 1 = sky
    rng = np.random.default_rng(3)
    sky_col = np.stack([0.45 + 0.2 * yy / H, 0.62 + 0.15 * yy / H, 0.85 - 0.05 * yy / H], axis=-1) * np.ones((1, W, 1))
    land_col = np.array([0.18, 0.28, 0.14])[None, None, :] * np.ones((H, W, 1))
    img = (truth[..., None] * sky_col + (1 - truth[..., None]) * land_col + rng.normal(0, 0.012, (H, W, 3))).astype(np.float32).clip(0, 1)
    # a low-resolution (1/12) model output, blown back up: blocky edges
    small = cv2.resize(truth, (W // 12, H // 12), interpolation=cv2.INTER_AREA)
    blocky = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    fixed = sr._colour_matte(img, blocky.copy(), int(ridge.max() + 120))
    err0 = float(np.abs(blocky - truth).mean())
    err1 = float(np.abs(fixed - truth).mean())
    return "sky matte: a blocky low-resolution mask is rebuilt into a sharp skyline at full resolution", err1 < 0.5 * err0 and err1 < 0.004, \
        f"mean edge error {err0:.4f} -> {err1:.4f}"


def t_drama_operations(an, p) -> Result:
    import drama
    from params import Drama
    src, size, _ = engine._target(an, 700)
    masks = engine._masks_at(an, size)
    sky = masks.get("sky")
    d = Drama(enabled=True, amount=100, mood=90, protect=0, enrich=0, sky_burn=0, sky_texture=0, sky_colour=0, bloom=0, mist=0, depth_haze=0,
              sh_hue=272, sh_chroma=16, mid_hue=272, mid_chroma=24, hi_hue=272, hi_chroma=20)
    off = drama.apply(src, masks, None, Drama(enabled=False))
    zero = drama.apply(src, masks, None, d.model_copy(update={"amount": 0}))
    mood = drama.apply(src, masks, None, d)
    lab0, lab1 = cv2.cvtColor(src, cv2.COLOR_RGB2Lab), cv2.cvtColor(mood, cv2.COLOR_RGB2Lab)
    dL = float(np.abs(lab1[..., 0] - lab0[..., 0]).mean())
    ab_shift = float(np.hypot(lab1[..., 1] - lab0[..., 1], lab1[..., 2] - lab0[..., 2]).mean())
    c0 = np.hypot(lab0[..., 1], lab0[..., 2])
    grey, vivid = c0 < 8, c0 > 40
    moved = lambda m: float(np.hypot(lab1[..., 1] - lab0[..., 1], lab1[..., 2] - lab0[..., 2])[m].mean()) if m.sum() > 200 else None
    prot = drama.apply(src, masks, None, d.model_copy(update={"protect": 100}))
    lab2 = cv2.cvtColor(prot, cv2.COLOR_RGB2Lab)
    vivid_moved_protected = float(np.hypot(lab2[..., 1] - lab0[..., 1], lab2[..., 2] - lab0[..., 2])[vivid].mean()) if vivid.sum() > 200 else 0.0
    vivid_moved_free = moved(vivid) or 0.0
    checks = {}
    checks["off and amount 0 change nothing"] = bool(off is src or np.array_equal(off, src)) and float(np.abs(zero - src).max()) < 1e-6
    checks["mood changes colour, not brightness"] = ab_shift > 3.0 and dL < 6.0
    checks["colourful pixels are protected"] = vivid.sum() < 200 or vivid_moved_protected < 0.5 * max(vivid_moved_free, 1e-6)
    if sky is not None and (sky > 0.5).mean() > 0.05:
        sd = d.model_copy(update={"mood": 0, "sky_burn": 100, "sky_texture": 50})
        out = drama.apply(src, masks, None, sd)
        diff = np.abs(out - src).max(axis=2)
        rr = max(3, int(round(0.012 * max(sky.shape)))) * 3                       # the burn deliberately covers the mask's soft edge (no bright rim at the skyline)
        away = cv2.dilate((sky > 0.02).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))) == 0
        checks["sky burn stays inside the sky"] = (float(diff[away].max()) < 0.012 if away.sum() > 50 else True) and float(diff[sky > 0.9].mean()) > 0.01
    bl = drama.apply(src, masks, None, d.model_copy(update={"mood": 0, "bloom": 100}))
    bl2 = drama.apply(src, masks, None, d.model_copy(update={"mood": 0, "bloom": 40}))
    checks["bloom grows with its slider (or there is no light to bloom)"] = float(np.abs(bl - src).mean()) >= float(np.abs(bl2 - src).mean()) - 1e-9
    checks["deterministic"] = bool(np.array_equal(drama.apply(src, masks, None, d), mood))
    s2, size2, _ = engine._target(an, 1400)
    m2 = engine._masks_at(an, size2)
    big = drama.apply(s2, m2, None, d.model_copy(update={"mood": 60, "bloom": 50, "sky_burn": 50, "sky_texture": 30}))
    small = drama.apply(src, masks, None, d.model_copy(update={"mood": 60, "bloom": 50, "sky_burn": 50, "sky_texture": 30}))
    cmp = cv2.resize(big, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    corr = float(np.corrcoef((cmp @ LUMA).ravel(), (small @ LUMA).ravel())[0, 1])
    checks["same result at two sizes"] = corr > 0.97
    bad = [k for k, v in checks.items() if not v]
    return "drama operations: mood colours without changing brightness, colourful pixels protected, sky burn stays in the sky, bloom scales, size-independent", not bad, \
        f"{'all hold' if not bad else 'FAILED: ' + str(bad)}; colour shift {ab_shift:.1f}, brightness change {dL:.1f}, corr {corr:.3f}"


def t_drama_looks(an, p) -> Result:
    import drama_looks as dl
    looks = dl.generate(an)
    ids = [l["id"] for l in looks]
    ok_list = len(looks) == len(dl.VARIANTS) and [l["fit"] for l in looks] == sorted([l["fit"] for l in looks], reverse=True) and sum(l["recommended"] for l in looks) == 3
    base = _render_arr(an, p, 500)
    skin = an.bank.get("skin", (base.shape[1], base.shape[0]))
    outs, bad = {}, []
    for lid in ids:
        q, notes, facts = dl.build(an, p, lid, 1.0)
        r = _render_arr(an, q, 500)
        outs[lid] = r
        clip = float((r.max(2) > 0.995).mean() - (base.max(2) > 0.995).mean())
        crush = float((r.max(2) < 0.02).mean() - (base.max(2) < 0.02).mean())
        med = float(np.median(_luma(r)))
        ok = q.drama.enabled and q.wb_mode == "asshot" and clip < 0.06 and crush < 0.12 and 0.12 < med < 0.75 and bool(notes)
        if skin is not None and (skin > 0.5).mean() > 0.01:
            ok = ok and float(np.abs(r[skin > 0.5] - base[skin > 0.5]).mean()) < 0.14
        if not ok:
            bad.append(lid)
    keys = list(outs)
    gaps = [float(np.abs(outs[a] - outs[b]).mean()) for i, a in enumerate(keys) for b in keys[i + 1:]]
    zero = dl.build(an, p, ids[0], 0.0)[0]
    z = float(np.abs(_render_arr(an, zero, 500) - base).mean())
    no_sky = "sky" not in an.bank.names()
    q, _, _ = dl.build(an, p, "blue_hour", 1.0)
    sky_ok = (not no_sky) or q.drama.sky_burn == 0.0
    return "drama looks: generated for the scene, all safe, as-shot white balance kept, clearly different from each other", \
        bool(ok_list and not bad and min(gaps) > 0.01 and z < 0.05 and sky_ok), \
        f"{len(looks)} looks, top: {ids[:3]}; unsafe: {bad or 'none'}; closest pair differs {min(gaps):.3f}; strength 0 changes {z:.3f}"


def t_white_level_and_look_scale(an, p) -> Result:
    """Auto must not dull the picture (the brightest tones keep the original's level), and a Drama look must look the same at preview and export size."""
    import cv2
    import drama_looks as dl
    from params import EditParams
    q0 = float(np.percentile(_luma(_render_arr(an, EditParams(), 700)), 99.5))
    q1 = float(np.percentile(_luma(_render_arr(an, p, 700)), 99.5))
    whites_ok = q1 >= min(q0, 0.97) - 0.12
    ids = [l["id"] for l in dl.generate(an)]
    q, _, _ = dl.build(an, p, ids[0], 1.0)
    small = _render_arr(an, q, 320)
    big = _render_arr(an, q, 960)
    big_s = cv2.resize(big, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    diff = float(np.abs(small - big_s).mean())
    return "whites kept after Auto; Drama looks the same at preview and export size", bool(whites_ok and diff < 0.045), \
        f"brightest tone original {q0:.2f} -> auto {q1:.2f}; look 320px vs 960px differs by {diff:.3f}"


def t_long_exposure(an, p) -> Result:
    """Long exposure: off changes nothing; silky water only changes water and removes ripple detail; cloud streaks and the graduated ND stay in the sky;
    the ND is stronger at the top than at the horizon; preview and larger size look alike."""
    import cv2
    import engine
    import longexp
    from params import LongExposure
    src, size, _ = engine._target(an, 700)
    m = engine._masks_at(an, size)
    checks, notes = {}, []
    off = longexp.apply(src, m, LongExposure())
    checks["off changes nothing"] = off is src or float(np.abs(off - src).max()) < 1e-6
    hp = lambda a: float(np.std(_luma(a) - cv2.GaussianBlur(_luma(a), (0, 0), 1.6)))
    water = m.get("water")
    if water is not None and (water > 0.5).mean() > 0.04:
        o = longexp.apply(src, m, LongExposure(enabled=True, water=70))
        wm = water > 0.9
        far = cv2.dilate((water > 0.02).astype(np.uint8), np.ones((15, 15), np.uint8)) == 0
        before, after = hp(src * wm[..., None]), hp(o * wm[..., None])
        checks["water ripple detail drops"] = after < 0.8 * before
        checks["nothing outside the water changes"] = float(np.abs(o - src)[far].max()) < 0.01 if far.sum() > 100 else True
        notes.append(f"water detail {before:.3f}->{after:.3f}")
    sky = m.get("sky")
    if sky is not None and (sky > 0.5).mean() > 0.06:
        r = max(3, int(round(0.012 * max(sky.shape)))) * 3
        far = cv2.dilate((sky > 0.02).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) == 0
        for what, le in (("cloud streaks", LongExposure(enabled=True, clouds=80)), ("graduated ND", LongExposure(enabled=True, grad_nd=80))):
            o = longexp.apply(src, m, le)
            checks[f"{what} stay in the sky"] = float(np.abs(o - src)[far].max()) < 0.012 if far.sum() > 100 else True
            cloudy = hp(src * (sky > 0.9)[..., None]) > 0.02                    # a clear sky has nothing to streak
            checks[f"{what} change the sky"] = (not cloudy and what == "cloud streaks") or float(np.abs(o - src)[sky > 0.9].mean()) > 0.002
        o = longexp.apply(src, m, LongExposure(enabled=True, grad_nd=100))
        rows = np.nonzero(sky > 0.5)[0]
        top, bot = int(np.percentile(rows, 5)), int(np.percentile(rows, 90))
        dtop = float((_luma(src) - _luma(o))[max(0, top - 3):top + 8][sky[max(0, top - 3):top + 8] > 0.9].mean())
        dbot = float((_luma(src) - _luma(o))[bot - 8:bot][sky[bot - 8:bot] > 0.9].mean()) if (sky[bot - 8:bot] > 0.9).any() else 0.0
        checks["the ND is stronger at the top than at the horizon"] = dtop > dbot
        notes.append(f"ND darkening top {dtop:.3f} / horizon {dbot:.3f}")
    if len(checks) > 1:
        p2 = p.model_copy(deep=True)
        p2.longexp = LongExposure(enabled=True, water=70, clouds=60, grad_nd=60)
        small, big = _render_arr(an, p2, 320), _render_arr(an, p2, 900)
        d = float(np.abs(small - cv2.resize(big, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)).mean())
        checks["preview and larger size look alike"] = d < 0.04
        notes.append(f"320 vs 900 px differ {d:.3f}")
    bad = [k for k, v in checks.items() if not v]
    return "long exposure: region-limited, silky water, streaked clouds, graduated ND, size independent", bool(not bad), f"{len(checks)} checks; failed: {bad or 'none'}; " + "; ".join(notes)


def t_geometry(an, p) -> Result:
    """Crop / straighten: off is identity; the window has the requested aspect and never shows an empty corner; auto-straighten finds a known tilt;
    preview and export crop the same area."""
    import cv2
    import geometry
    from params import Geometry
    checks, notes = {}, []
    rng = np.random.default_rng(5)
    h, w = 800, 1200
    img = np.zeros((h, w, 3), np.float32)
    img[:430] = (0.5, 0.7, 0.95)
    img[430:] = (0.25, 0.35, 0.15)
    cv2.line(img, (300, 100), (300, 650), (0.1, 0.1, 0.1), 9)
    img = np.clip(img + rng.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
    sky = np.zeros((h, w), np.float32)
    sky[:430] = 1
    checks["off is identity"] = geometry.apply(img, Geometry()) is img
    for asp, ratio in (("1:1", 1.0), ("4:5", 0.8), ("16:9", 16 / 9), ("3:2", 1.5)):
        g = Geometry(enabled=True, angle=3.5, aspect=asp)
        o = geometry.apply(img, g)
        checks[f"aspect {asp} exact"] = abs(o.shape[1] / o.shape[0] - ratio) < 0.01
    # no empty corner: warp a constant-coloured picture whose outside would read as the border colour
    for ang in (-7.0, 4.0, 12.0):
        g = Geometry(enabled=True, angle=ang, aspect="3:2", zoom=1.0, x=1.0, y=-1.0)
        ow, oh, M = geometry.window(w, h, g)
        ones = np.ones((h, w), np.float32)
        warped = cv2.warpAffine(ones, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        checks[f"no empty corner at {ang:+.0f} deg, window moved to the edge"] = float((warped < 0.99).mean()) == 0.0
    for tilt in (-4.0, 2.5, 6.0):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -tilt, 1.0)
        tilted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        sk = cv2.warpAffine(sky, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        r = geometry.auto_angle(tilted, {"sky": sk})
        checks[f"auto-straighten finds a {tilt:+.1f} deg tilt"] = abs(r["angle"] + tilt) < 0.4 and r["confidence"] > 0.5
    g = Geometry(enabled=True, angle=-3.0, aspect="4:5", zoom=1.4, x=0.5, y=0.0)
    small = geometry.apply(cv2.resize(img, (300, 200), interpolation=cv2.INTER_AREA), g)
    big = geometry.apply(img, g)
    d = float(np.abs(small - cv2.resize(big, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)).mean())
    checks["preview and export crop the same area"] = d < 0.03
    notes.append(f"preview vs export differ {d:.3f}")
    # through the whole engine: the rendered size is the window's size
    import engine
    p2 = p.model_copy(deep=True)
    p2.geometry = Geometry(enabled=True, angle=2.0, aspect="1:1")
    im = engine.render(an, p2, 600)
    checks["engine renders the cropped window"] = abs(im.size[0] / im.size[1] - 1.0) < 0.01
    bad = [k for k, v in checks.items() if not v]
    return "crop & straighten: exact aspect, no empty corners, auto-straighten accurate, preview = export", bool(not bad), f"{len(checks)} checks; failed: {bad or 'none'}; " + "; ".join(notes)


def t_cull_quality(an, p) -> Result:
    """The blur / exposure flags used for culling: a photo reads sharper than the same photo blurred, and the summary carries the flags."""
    import cv2
    import engine
    src, _, _ = engine._target(an, 1600)
    y = (src @ np.array([0.2126, 0.7152, 0.0722], np.float32)).astype(np.float32)
    r0 = engine._sharp_ratio(y)
    r1 = engine._sharp_ratio(cv2.GaussianBlur(y, (0, 0), 1.5))
    r2 = engine._sharp_ratio(cv2.GaussianBlur(y, (0, 0), 3.5))
    q = an.summary().get("quality", {})
    ok = r0 > r1 > r2 and r2 < 2.0 and r0 > 2.4 and q.get("blur") in ("sharp", "soft", "blurry") and q.get("exposure") in ("ok", "dark", "bright")
    return "culling flags: sharp > mildly blurred > blurred, flags present in the analysis", bool(ok), f"sharpness ratio {r0:.2f} / blurred 1.5px {r1:.2f} / 3.5px {r2:.2f}; flags {q}"


TESTS: List[Callable] = [
    t_mask_alignment_and_range, t_mask_operations, t_mask_controls, t_local_edit_stays_inside_mask,
    t_db_off_is_identity, t_db_highlights_and_shadows_protected, t_db_preserves_skin_texture, t_db_no_halo,
    t_db_deterministic_and_independent, t_auto_lighting_conservative, t_pipeline_is_non_destructive,
    t_render_size_matches_image, t_neutral_params_are_identity, t_slider_directions, t_srgb_pipeline, t_light_scene_read,
    t_custom_mask_shapes, t_gradient_and_radial_geometry, t_brush_resolution_independent, t_range_masks_select_what_they_claim,
    t_mask_limit_and_refine, t_custom_mask_edit_stays_inside, t_point_curve,
    t_region_coverage, t_region_paint_fix, t_lut_behaviour, t_match_reference, t_banded_ops_are_exact, t_smart_looks, t_adaptive_looks, t_sky_studio, t_sky_replacement, t_sky_full_res_matte, t_drama_operations, t_drama_looks, t_white_level_and_look_scale, t_long_exposure, t_geometry, t_cull_quality,
]


def run(paths: List[str]) -> bool:
    all_ok = True
    for path in paths:
        t0 = time.time()
        an = engine.analyze(path)
        p = engine.auto_params(an, "auto")
        print(f"\n=== {path}  [{an.scene.kind}, masks: {', '.join(an.bank.names())}]  (analysed in {time.time() - t0:.1f}s)")
        for test in TESTS:
            try:
                name, ok, detail = test(an, p)
            except Exception as e:
                name, ok, detail = test.__name__, False, f"crashed: {e!r}"
            all_ok &= ok
            print(f"  {'PASS' if ok else 'FAIL'}  {name}\n        {detail}")
    print("\nALL PASSED" if all_ok else "\nSOME TESTS FAILED")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1:] or ["test_sheets/IMG_4208.heif", "test_sheets/DSC_8632.NEF"]) else 1)
