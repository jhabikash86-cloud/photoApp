"""Adaptive sky replacement.

Library: the free CC0 "Pure Skies" panoramas from Poly Haven (see fetch_skies.py), 360-degree equirectangular skies with
metadata (time of day, weather, contrast, season).

What makes a replacement look real, and how each point is handled here:
  1. The new sky must suit the light the landscape was actually shot in  -> `rank()` scores every sky against the photo's
     light, weather, tonal range and colour; a mismatched sky is ranked low and flagged.
  2. It must look like it was photographed with the same lens                -> the panorama is projected through a pinhole
     camera (the photo's field of view), with the horizon placed where this photo's sky ends.
  3. No halo of the old sky around trees, hills, fences                     -> the sky mask is snapped to the picture's real
     edges and the foreground edge colours are decontaminated (the old sky's colour is removed from them).
  4. The horizon must stay believable                                       -> the original horizon haze is blended into the
     bottom of the new sky, so distant hills keep their atmosphere.
  5. The landscape must belong under the new sky                            -> it is gently relit (colour + brightness) toward
     the new sky's light; water takes on the new sky's colour (reflections), by an amount you control.
  6. Composition                                                            -> azimuth / height are chosen automatically (sun kept
     out of frame, the most interesting clouds in view) and stay adjustable.

`plan()` makes every adaptive decision once, when a sky is picked, and stores the result in EditParams.sky_replace. The render then
only projects and composites, so it is cheap and every number remains editable.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from params import SkyReplace

Image.MAX_IMAGE_PIXELS = None
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
SKY_DIR = Path(__file__).parent / "skies"
CACHE = SKY_DIR / ".cache"
BASE_HFOV = 72.0                      # degrees: a typical 24-26 mm (35 mm equivalent) landscape / phone wide lens
LIGHT_BUCKET = {"sunny": "day", "overcast": "day", "golden_hour": "low", "sunset": "low", "blue_hour": "dusk", "night": "night", "indoor": "day"}
TIME_BUCKET = {"morning": "day", "midday": "day", "afternoon": "day", "sunrise": "low", "sunset": "low", "dusk": "dusk", "night": "night"}
TIME_LABEL = {"morning": "Morning", "midday": "Midday", "afternoon": "Afternoon", "sunrise": "Sunrise", "sunset": "Sunset", "dusk": "Dusk", "night": "Night"}
WEATHER_LABEL = {"clear": "clear", "partly_cloudy": "partly cloudy", "overcast": "overcast", "fog": "misty"}


# ------------------------------------------------------------------------------------------------ library
_index: Optional[Dict[str, dict]] = None


def library() -> Dict[str, dict]:
    """id -> info, from skies/index.json. Empty if the sky pack has not been downloaded."""
    global _index
    if _index is None:
        _index = {}
        f = SKY_DIR / "index.json"
        if f.exists():
            meta = json.loads(f.read_text())
            for sid, m in meta.items():
                if not (SKY_DIR / f"{sid}.jpg").exists():
                    continue
                a = m.get("attributes", {})
                _index[sid] = {"id": sid, "name": m.get("name", sid).replace(" (Pure Sky)", ""), "time": a.get("time_of_day", "midday"),
                               "weather": a.get("weather", "clear"), "contrast": a.get("contrast", "medium"), "season": a.get("season"),
                               "kelvin": m.get("whitebalance"), "obstructed": a.get("sky_view") == "obstructed"}
    return _index


def label(info: dict) -> str:
    return f"{TIME_LABEL.get(info['time'], info['time'])}, {WEATHER_LABEL.get(info['weather'], info['weather'])}" + (f", {info['season']}" if info.get("season") else "")


def _plate_source(sid: str, need_w: int) -> np.ndarray:
    """The upper hemisphere (plus a little mirrored margin) of a sky panorama, uint8 RGB, at least `need_w` wide when possible.
    Previews use a cached 4096-wide copy; exports read the full-resolution file."""
    CACHE.mkdir(exist_ok=True)
    if need_w <= 4096:
        c = CACHE / f"{sid}_4k.jpg"
        if not c.exists():
            im = Image.open(SKY_DIR / f"{sid}.jpg").convert("RGB")
            im = im.resize((4096, 2048), Image.LANCZOS).crop((0, 0, 4096, 1100))
            im.save(c, "JPEG", quality=92)
        return np.asarray(Image.open(c).convert("RGB"))
    im = Image.open(SKY_DIR / f"{sid}.jpg").convert("RGB")
    w, h = im.size
    return np.asarray(im.crop((0, 0, w, int(h * 0.537))))


def project(sid: str, size: Tuple[int, int], rows: int, horizon_y: float, hfov: float, azimuth: float, lift: float) -> np.ndarray:
    """The panorama as a photograph: float32 RGB, `rows` rows tall and size[0] wide (the top of the frame). The horizon (elevation 0)
    sits on row `horizon_y` of a frame of size[1] rows; hfov is the horizontal field of view in degrees."""
    W, H = size
    f = (W / 2) / math.tan(math.radians(hfov) / 2)
    cy = H / 2
    pitch = math.atan2(horizon_y - cy, f) + math.radians(lift)
    ppd_needed = W / hfov
    src = _plate_source(sid, int(ppd_needed * 360 * 1.0))
    sh, sw = src.shape[:2]
    out = np.empty((rows, W, 3), np.float32)
    cp, sp = math.cos(pitch), math.sin(pitch)
    xs = (np.arange(W, dtype=np.float32) - W / 2 + 0.5) / f
    for y0 in range(0, rows, 512):
        y1 = min(rows, y0 + 512)
        ys = (cy - (np.arange(y0, y1, dtype=np.float32) + 0.5)) / f
        X, Y = np.meshgrid(xs, ys)
        # ray through the pixel, tilted by the camera pitch
        y_t = Y * cp + sp
        z_t = -Y * sp + cp
        lon = np.arctan2(X, z_t) + math.radians(azimuth)
        el = np.arctan2(y_t, np.sqrt(X * X + z_t * z_t))
        u = ((lon / (2 * math.pi) + 0.5) % 1.0) * sw
        v = (0.5 - el / math.pi) * (sw / 2)
        # below the horizon the panorama is a mirror image; keep inside the stored rows
        v = np.clip(v, 0, sh - 1)
        out[y0:y1] = cv2.remap(src, u.astype(np.float32), v.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP).astype(np.float32) / 255.0
    return np.clip(out, 0, 1)


def _sun_azimuth(sid: str) -> Tuple[Optional[float], float]:
    """(azimuth of the brightest spot in degrees, its elevation) from a small copy of the panorama."""
    a = _plate_source(sid, 4096)
    s = cv2.resize(a, (1024, int(1024 * a.shape[0] / a.shape[1])), interpolation=cv2.INTER_AREA)
    y = cv2.GaussianBlur((s.astype(np.float32) / 255 @ LUMA), (0, 0), 6)
    iy, ix = np.unravel_index(int(np.argmax(y)), y.shape)
    el = 90.0 - iy / (s.shape[1] / 2) * 180.0
    if el < 2 or y.max() < 0.8:
        return None, el
    return (ix / s.shape[1] - 0.5) * 360.0, el


# ------------------------------------------------------------------------------------------------ descriptors and thumbnails
_desc: Optional[Dict[str, dict]] = None


def descriptors() -> Dict[str, dict]:
    """Measured properties of each sky (cached in skies/.cache/desc.json): cloud detail, brightness, warmth, sun position."""
    global _desc
    if _desc is None:
        CACHE.mkdir(exist_ok=True)
        f = CACHE / "desc.json"
        _desc = json.loads(f.read_text()) if f.exists() else {}
        todo = [sid for sid in library() if sid not in _desc]
        for sid in todo:
            a = _plate_source(sid, 4096).astype(np.float32) / 255
            h, w = a.shape[:2]
            band = a[int(h * 0.22):int(h * 0.46)]                       # elevation ~ 20..60 degrees
            y = band @ LUMA
            warm = float(np.median(band[..., 0]) / max(np.median(band[..., 2]), 1e-3))
            az, el = _sun_azimuth(sid)
            _desc[sid] = {"interest": float(np.std(y - cv2.GaussianBlur(y, (0, 0), 25))), "luma": float(np.median(y)), "warm": warm,
                          "sun_az": az, "sun_el": float(el)}
        if todo:
            f.write_text(json.dumps(_desc))
    return _desc


def thumb_path(sid: str) -> Path:
    """A small preview of the sky as a photograph would see it (sun out of frame). Created on first use."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{sid}_thumb.jpg"
    if not f.exists():
        az = descriptors().get(sid, {}).get("sun_az")
        az = 0.0 if az is None else ((az + BASE_HFOV / 2 + 28 + 180) % 360) - 180
        pl = project(sid, (480, 300), 300, 270, BASE_HFOV, az, 8.0)
        Image.fromarray((pl * 255).astype(np.uint8)).save(f, "JPEG", quality=85)
    return f


# ------------------------------------------------------------------------------------------------ ranking
def rank(an, facts_light: Optional[str], tonal) -> List[dict]:
    """Every library sky scored against this photo. fit 0..1 with the reasons."""
    import sky_engine
    lib = library()
    if not lib:
        return []
    desc = descriptors()
    r = sky_engine.read_sky(an)
    photo_bucket = LIGHT_BUCKET.get(facts_light or "sunny", "day")
    hard = facts_light in ("sunny", "golden_hour", "sunset")
    out = []
    for sid, info in lib.items():
        reasons = []
        tb = TIME_BUCKET.get(info["time"], "day")
        if tb == photo_bucket:
            t = 1.0
        elif {tb, photo_bucket} == {"day", "low"}:
            t = 0.30
        elif {tb, photo_bucket} == {"low", "dusk"}:
            t = 0.7
        elif "night" in (tb, photo_bucket):
            t = 0.04
        else:
            t = 0.2
        if t >= 0.9:
            reasons.append("same time of day as the landscape's light")
        elif t < 0.4:
            reasons.append("different time of day: the landscape's light would not match")
        w = info["weather"]
        if facts_light == "overcast":
            wf = {"overcast": 1.0, "fog": 0.9, "partly_cloudy": 0.5, "clear": 0.15}.get(w, 0.4)
        elif hard:
            wf = {"clear": 1.0, "partly_cloudy": 1.0, "overcast": 0.15, "fog": 0.1}.get(w, 0.4)
        else:
            wf = {"clear": 0.8, "partly_cloudy": 0.9, "overcast": 0.6, "fog": 0.4}.get(w, 0.5)
        if wf < 0.4:
            reasons.append("weather does not match the landscape's lighting")
        c = {"low": 0.0, "medium": 0.5, "high": 1.0}.get(info["contrast"], 0.5)
        hard_fg = float(np.clip((tonal.rng - 0.5) / 0.35, 0, 1)) if tonal is not None else 0.5
        cf = 1.0 - 0.35 * abs(c - hard_fg)
        d = desc.get(sid, {})
        # a flat or blown original gains most from a sky with real cloud detail; warmth should suit the light
        want_detail = float(np.clip((0.30 - (r.rng if r is not None else 0.3)) / 0.2, 0, 1)) if r is not None else 0.3
        interest = float(np.clip(d.get("interest", 0.05) / 0.12, 0, 1))
        kref = {"day": 6000, "low": 4300, "dusk": 6500, "night": 9000}[photo_bucket]
        kf = 1.0 - 0.15 * min(1.0, abs(math.log((info.get("kelvin") or kref) / kref)) * 2)
        fit = float(t * wf * cf * kf) * (1.0 + 0.10 * want_detail * (interest - 0.5))
        if info["obstructed"]:
            fit *= 0.4
        out.append({"id": sid, "name": info["name"], "label": label(info), "fit": round(min(1.0, fit), 2), "reason": "; ".join(reasons) or "suits the light", "time": info["time"], "weather": info["weather"]})
    out.sort(key=lambda d: -d["fit"])
    return out


def need(an) -> dict:
    """Would this photo benefit from a new sky? {needed, reason}. Replacement is offered always, recommended only when it helps."""
    import sky_engine
    r = sky_engine.read_sky(an)
    size = (an.work.shape[1], an.work.shape[0])
    water = float((an.bank.get("water", size) > 0.5).mean()) if "water" in an.bank.names() else 0.0
    if r is None:
        return {"available": False, "needed": False, "reason": "No sky found in this photo."}
    why = []
    if r.clip > 0.04:
        why.append(f"{r.clip * 100:.0f}% of the sky is blown out")
    if r.kind == "overcast" and r.rng < 0.22:
        why.append("flat, featureless grey sky")
    if r.kind == "hazy":
        why.append("washed-out hazy sky")
    out = {"available": True, "needed": bool(why), "reason": "; ".join(why) or f"the sky is already {sky_engine.KIND_LABEL[r.kind].lower()}: enhancing it (Sky tab) usually looks more natural than replacing it"}
    if water > 0.06:
        out["warning"] = f"{water * 100:.0f}% of the photo is water: reflections are only colour-matched, not redrawn, so a very different sky may look off"
    return out


# ------------------------------------------------------------------------------------------------ geometry of this photo
def _lowest(m: np.ndarray, pct: float = 100.0) -> float:
    """Row of the lowest sky pixel (or a high percentile of sky rows)."""
    rows = np.nonzero(m > 0.5)[0]
    return float(np.percentile(rows, pct)) if len(rows) else m.shape[0] * 0.4


def _horizon(sky_mask: np.ndarray) -> float:
    ys = np.nonzero((sky_mask > 0.5).any(axis=1))[0]
    return float(np.percentile(np.nonzero(sky_mask > 0.5)[0], 97)) if len(ys) else sky_mask.shape[0] * 0.4


def _interest(plate: np.ndarray) -> float:
    """How much cloud detail is in view (high-pass luma energy)."""
    y = plate @ LUMA
    return float(np.std(y - cv2.GaussianBlur(y, (0, 0), max(2.0, y.shape[1] / 40))))


def plan(an, sky_id: str) -> SkyReplace:
    """Every adaptive decision for putting sky `sky_id` into this photo: composition, brightness, horizon haze, relight."""
    lib = library()
    if sky_id not in lib:
        raise ValueError("That sky is not in the library.")
    size = (an.work.shape[1], an.work.shape[0])
    sky = an.bank.get("sky", size)
    if sky is None or (sky > 0.5).mean() < 0.02:
        raise ValueError("No sky was found in this photo.")
    W, H = size
    hy = _lowest(sky, 99.0)
    rows = int(min(H, _lowest(sky) + 6))
    # --- composition: keep the sun out of frame, show the most interesting clouds
    sun_az, sun_el = _sun_azimuth(sky_id)
    cands = [0.0, 90.0, 180.0, -90.0]
    if sun_az is not None:
        off = BASE_HFOV / 2 + 28
        cands = [sun_az + off, sun_az - off, sun_az + off + 70, sun_az - off - 70]
    best, best_s = (0.0, 0.0), -1.0
    for az in cands:
        az = ((az + 180) % 360) - 180
        for lift in (0.0, 10.0):
            pl = project(sky_id, (min(W, 640), int(H * min(W, 640) / W)), int(rows * min(W, 640) / W), hy * min(W, 640) / W, BASE_HFOV, az, lift)
            s = _interest(pl)
            if sun_az is not None:
                d = abs(((sun_az - az + 180) % 360) - 180)
                if d < BASE_HFOV / 2 + 6 and sun_el < 60:
                    s *= 0.3
            if s > best_s:
                best, best_s = (az, lift), s
    az, lift = best
    # --- brightness: match the new sky's horizon band to the original's (never brighter than a believable sky)
    sw = 640
    sz = (sw, int(H * sw / W))
    rows_s = int(rows * sw / W)
    pl = project(sky_id, sz, rows_s, hy * sw / W, BASE_HFOV, az, lift)
    sky_s = cv2.resize(sky, (sz[0], rows_s), interpolation=cv2.INTER_AREA)
    orig = cv2.resize(an.work, (sz[0], sz[1]), interpolation=cv2.INTER_AREA)[:rows_s]
    band = np.zeros_like(sky_s, bool)
    top = int(rows_s * 0.55)
    band[top:] = sky_s[top:] > 0.7
    if band.sum() < 50:
        band = sky_s > 0.7
    yo = float(np.median((orig @ LUMA)[band])) if band.sum() else 0.6
    yn = float(np.median((pl @ LUMA)[band])) if band.sum() else 0.6
    yo = min(yo, 0.82)
    ratio = float(np.clip((yo + 1e-3) / (yn + 1e-3), 0.55, 1.6))
    exposure = float(np.clip(math.log2(ratio) / 1.5 * 100 * 0.7, -60, 60))    # 70% of the way: keeps the new sky's own character
    # --- haze, relight, reflections: gentler when the new sky is a poor match for the light
    t = need(an)
    water = float((an.bank.get("water", size) > 0.5).mean()) if "water" in an.bank.names() else 0.0
    return SkyReplace(id=sky_id, blend=100, azimuth=float(az), lift=float(lift), zoom=100, exposure=exposure, horizon=45, relight=55,
                      reflect=float(70 if water > 0.02 else 0), edge=60)


# ------------------------------------------------------------------------------------------------ the composite
def _smooth(a, e0, e1):
    t = np.clip((a - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _horizon_colour(src: np.ndarray, sky_a: np.ndarray, rows: int) -> np.ndarray:
    """A colour per column: the original sky's haze just above the skyline (blurred sideways), for blending into the new sky."""
    H, W = sky_a.shape
    s = min(1.0, 700 / max(H, W))
    sm = cv2.resize(src[:rows], None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else src[:rows]
    am = cv2.resize(sky_a[:rows], (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_AREA) if s < 1 else sky_a[:rows]
    h = sm.shape[0]
    band = np.zeros_like(am)
    # the lowest sky pixels of each column (the band touching the skyline)
    idx = np.where(am > 0.9, np.arange(h)[:, None], -1).max(axis=0)
    for x in range(am.shape[1]):
        if idx[x] > 0:
            band[max(0, idx[x] - max(2, h // 14)):idx[x] + 1, x] = 1.0
    band *= am > 0.8
    sig = max(3.0, sm.shape[1] / 10)
    num = cv2.GaussianBlur((sm * band[..., None]).sum(axis=0, keepdims=True).astype(np.float32), (0, 0), sig, borderType=cv2.BORDER_REPLICATE)
    den = cv2.GaussianBlur(band.sum(axis=0, keepdims=True).astype(np.float32), (0, 0), sig, borderType=cv2.BORDER_REPLICATE)
    glob = np.median(sm[band > 0.5], axis=0) if (band > 0.5).any() else np.array([0.8, 0.82, 0.85], np.float32)
    col = np.where(den[..., None] > 0.5, num / np.maximum(den[..., None], 1e-3), glob)[0]
    col = cv2.resize(col[None].astype(np.float32), (W, 1), interpolation=cv2.INTER_LINEAR)[0]
    return col.astype(np.float32)


def _colour_matte(src: np.ndarray, m: np.ndarray, rows: int) -> np.ndarray:
    """Sharpen a blocky low-resolution sky mask at full resolution. Near the skyline, each pixel is judged by its OWN colour: how
    much it looks like the sky just next to it versus the landscape just next to it (local colour models, per stretch of the
    skyline). Where sky and distant hills are nearly the same colour the model cannot tell them apart, so the original mask is kept."""
    H, W = m.shape
    k = max(6, max(H, W) // 220)
    hard = (m[:rows] > 0.5).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    band = (cv2.dilate(hard, ker) - cv2.erode(hard, ker)).astype(bool)
    if band.sum() < 500:
        return m
    near = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6 * k + 1, 6 * k + 1))) - cv2.erode(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6 * k + 1, 6 * k + 1)))
    lab = cv2.cvtColor(np.clip(src[:rows], 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    out = m.copy()
    mm = m[:rows]
    n = 16
    for c in range(n):
        x0, x1 = c * W // n, (c + 1) * W // n
        pad = W // (2 * n)
        xs0, xs1 = max(0, x0 - pad), min(W, x1 + pad)
        sel = near[:, xs0:xs1].astype(bool)
        sky_s = lab[:, xs0:xs1][sel & (mm[:, xs0:xs1] > 0.95)]
        fg_s = lab[:, xs0:xs1][sel & (mm[:, xs0:xs1] < 0.05)]
        if len(sky_s) < 300 or len(fg_s) < 300:
            continue
        mu_s, mu_f = sky_s.mean(0), fg_s.mean(0)
        sep = float(np.linalg.norm(mu_s - mu_f))
        w = float(np.clip((sep - 7) / 15, 0, 1))
        if w <= 0:
            continue
        reg = np.eye(3, dtype=np.float32) * 4.0
        ic_s = np.linalg.inv(np.cov(sky_s.T) + reg)
        ic_f = np.linalg.inv(np.cov(fg_s.T) + reg)
        bsel = band[:, x0:x1]
        px = lab[:, x0:x1][bsel]
        ds = np.einsum("ij,jk,ik->i", px - mu_s, ic_s, px - mu_s)
        df = np.einsum("ij,jk,ik->i", px - mu_f, ic_f, px - mu_f)
        alpha = 1.0 / (1.0 + np.exp(np.clip((np.sqrt(ds) - np.sqrt(df)) * 1.6, -30, 30)))
        cur = mm[:, x0:x1][bsel]
        out[:rows, x0:x1][bsel] = cur * (1 - w) + alpha * w
    return out


def apply(an, src: np.ndarray, masks: Dict[str, np.ndarray], sr: SkyReplace) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """The photo with the sky replaced. Returns (picture, refined sky mask). `masks` are at the picture's size."""
    import custom_masks
    import matting
    sky = masks.get("sky")
    if not sr.id or sr.blend <= 0 or sky is None or sr.id not in library():
        return src, None
    H, W = src.shape[:2]
    edge = sr.edge / 100.0
    m = matting.clean_mask(sky.astype(np.float32), max(H, W))
    m = custom_masks._guide(src, m, 35 + 55 * edge)                      # snap the sky edge to the picture's real edges
    if max(H, W) > 1400:
        # the mask comes from a low-resolution model: at export size its edge is blocky. Rebuild it from the picture's own colours at full
        # resolution near the skyline, then snap it to the real edges.
        from landscape import _guided
        r0 = int(min(H, _lowest(m) + max(H, W) // 20))
        m = _colour_matte(src, m, r0)
        g = cv2.cvtColor(src[:r0], cv2.COLOR_RGB2GRAY)
        m[:r0] = np.clip(_guided(g, m[:r0].copy(), max(3, max(H, W) // 900), 1e-4), 0, 1)
    m = np.clip((m - 0.06 * edge) / (1 - 0.12 * edge), 0, 1).astype(np.float32)   # push edge bleed to the extremes
    hy = _lowest(m, 99.0)
    rows = int(min(H, _lowest(m) + 6))                                  # down to the lowest sky pixel (valleys!), not just the typical skyline
    plate = project(sr.id, (W, H), rows, hy, BASE_HFOV / (sr.zoom / 100.0), sr.azimuth, sr.lift)
    # brightness (linear light) of the new sky
    if abs(sr.exposure) > 0.05:
        plate = np.clip(np.power(np.power(plate, 2.2) * 2.0 ** (sr.exposure / 100 * 1.5), 1 / 2.2), 0, 1)
    a = (m[:rows] * (sr.blend / 100.0)).astype(np.float32)                 # how much of each pixel is now new sky
    # original horizon haze continues into the new sky
    if sr.horizon > 0:
        hc = _horizon_colour(src, m, rows)
        ramp = _smooth(np.arange(rows, dtype=np.float32), hy - 0.30 * hy, hy)[:, None] * (sr.horizon / 100.0) * 0.6
        plate = plate * (1 - ramp[..., None]) + hc[None, :, :] * ramp[..., None]
    fg = src
    if edge > 0:                                                            # remove the old sky's colour from edge pixels
        alpha_fg = 1.0 - m
        sure = cv2.erode((alpha_fg > 0.98).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        clean = matting.decontaminate(src, alpha_fg, sure)
        fg = src * (1 - edge) + clean * edge
    out = fg.copy()
    out[:rows] = fg[:rows] * (1 - a[..., None]) + plate * a[..., None]
    # relight: the landscape takes on the new sky's light; water takes on its colour
    if sr.relight > 0 or sr.reflect > 0:
        sel = a > 0.85
        old = src[:rows][sel] if sel.any() else None
        if old is not None and len(old) > 100:
            newc, oldc = np.median(plate[sel], axis=0), np.median(old, axis=0)
            nl, ol = float(newc @ LUMA), float(oldc @ LUMA)
            tint = (newc / max(nl, 1e-3)) / (oldc / max(ol, 1e-3))           # new sky's colour cast relative to the old one's
            bright = (nl / max(ol, 1e-3))
            k = sr.relight / 100.0
            gain = np.clip(tint ** (0.35 * k), 0.92, 1.08) * float(np.clip(bright ** (0.12 * k), 0.92, 1.08))
            land = (1.0 - a_full(a, H, W, rows))[..., None]
            out = out * (1 + (gain - 1) * land)
            w = masks.get("water")
            if w is not None and sr.reflect > 0:
                kr = sr.reflect / 100.0
                wg = np.clip(tint ** (0.9 * kr), 0.72, 1.4) * float(np.clip(bright ** (0.3 * kr), 0.8, 1.25))
                # mix toward the new sky's colour, strongest where the old sky was reflected (a weak 'sky' response in the water)
                wm = np.clip(w * (0.5 + 0.5 * np.clip(sky * 4, 0, 1)), 0, 1)[..., None] * kr
                out = out * (1 + (wg - 1) * wm)
    return np.clip(out, 0, 1).astype(np.float32), m


def a_full(a: np.ndarray, H: int, W: int, rows: int) -> np.ndarray:
    """The 'now new sky' amount for the whole frame (rows below the plate are all landscape)."""
    full = np.zeros((H, W), np.float32)
    full[:rows] = a
    return full
