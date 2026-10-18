"""Dodge & burn engine: mask-aware, gentle, texture-preserving tonal lighting.

How it works
  * Every adjustment is a *smooth exposure map* (in EV), built from masks and tonal weights, then applied as a
    multiplicative gain in linear light, the way a darkroom dodge/burn works. Because the gain is low-frequency,
    skin pores, hair and fabric keep exactly the texture they had: nothing is sharpened, blurred or invented.
  * Maps are feathered and re-anchored to the picture's edges (guided filter), so there are no halos or visible
    retouching boundaries around hair, shoulders or ridges.
  * Dodge is protected against clipped highlights (headroom fade + soft shoulder); burn is protected against
    crushed shadows.
  * Nothing here moves pixels: no facial geometry changes, no invented features.

Layers (each one independently adjustable)
  dodge       brighten the chosen tones of the subject (skin when a skin mask exists, else the person, else a
              conservative global lift)
  burn        darken the chosen tones, by default in the background, never on skin when `protect_face`
  separation  darker background (stronger toward the frame edges) + a slightly lifted subject
  form        emphasise the light and shadow that is already on the face/skin (low-frequency luminance only)

`auto_lighting` analyses the picture and proposes conservative values for these sliders; the user can reduce,
increase, disable or reset them. With no reliable subject it only proposes gentle global corrections.
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from adjust import edge_aware_blur
from landscape import _guided
from params import DodgeBurn

GAMMA = 2.2
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
DODGE_EV = 0.75     # EV at slider 100
BURN_EV = 1.2
SEP_BURN_EV = 0.7
SEP_LIFT_EV = 0.25
FORM_EV = 0.65      # sculpting range at slider 100
GEO_EV = 0.28       # extra structural weight, used only where the light is flat
WORK_EDGE = 800     # maps are smooth, so they are computed at this size and upsampled


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _tone_weight(y: np.ndarray, focus: float) -> np.ndarray:
    """Soft weight, 1 at the chosen tone (focus 0..100 = shadows..highlights) falling off smoothly."""
    c = focus / 100.0
    return np.exp(-(((y - c) / 0.30) ** 2)).astype(np.float32)


def _small(a: np.ndarray, s: float) -> np.ndarray:
    return cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else a


def _reliable(m: Optional[np.ndarray], min_area: float) -> bool:
    return m is not None and float((m > 0.5).mean()) >= min_area


def compute_ev(arr: np.ndarray, masks: Dict[str, np.ndarray], p: DodgeBurn) -> Optional[np.ndarray]:
    """The exposure map (EV, float32, same size as arr) for the current parameters, or None if nothing to do."""
    if not p.enabled or not any((p.dodge, p.burn, p.separation, p.form)) or p.strength <= 0:
        return None
    H, W = arr.shape[:2]
    edge = max(H, W)
    s = min(1.0, WORK_EDGE / edge)
    a = _small(arr, s)
    h, w = a.shape[:2]
    y = (a @ LUMA).astype(np.float32)
    yb = np.clip(edge_aware_blur(y, max(h, w) / 40, 0.03), 0, 1)           # smooth luminance: tonal ranges

    def mk(name: str) -> Optional[np.ndarray]:
        m = masks.get(name)
        return None if m is None else _small(m.astype(np.float32), s)

    skin, person = mk("skin"), mk("person")
    fg, bg = mk("foreground"), mk("background")
    skin_ok = _reliable(skin, 0.002)
    person_ok = _reliable(person, 0.02) or _reliable(fg, 0.02)
    subject = person if _reliable(person, 0.02) else fg
    if bg is None and subject is not None:
        bg = 1.0 - subject

    # The softness follows the SUBJECT's size (a small face needs a small feather, a full figure a wide one).
    if skin_ok:
        ref = float(np.sqrt((skin > 0.5).sum()))
    elif person_ok:
        ref = float(np.sqrt((subject > 0.5).sum()))
    else:
        ref = 0.3 * max(h, w)
    soft = max(2.0, ref * (0.03 + 0.12 * p.feather / 100))
    skin_m = np.clip(skin, 0, 1) if skin_ok else None
    # the subject's lift uses only the solid core of the mask: its soft fringe still contains background
    subj_m = _smoothstep(0.40, 0.90, np.clip(subject, 0, 1)) if person_ok else None
    bg_m = np.clip(bg, 0, 1) if (person_ok and bg is not None) else None
    face_guard = None
    if p.protect_face and skin_ok:
        k = max(3, int(max(h, w) * 0.01)) | 1
        face_guard = cv2.GaussianBlur(cv2.dilate(skin_m, np.ones((k, k), np.uint8)), (0, 0), soft)

    # Each adjustment is accumulated in the REGION it belongs to (skin / subject / background / whole frame) and
    # later smoothed inside that region only. That keeps every edge attached to the subject: no light aura where
    # a blurred mask would leak across the outline, and no dark fringe on the subject.
    ev_skin = np.zeros((h, w), np.float32)
    ev_subj = np.zeros((h, w), np.float32)
    ev_bg = np.zeros((h, w), np.float32)
    ev_glob = np.zeros((h, w), np.float32)
    ones = np.ones((h, w), np.float32)

    # ---- dodge -----------------------------------------------------------------------------------
    if p.dodge > 0:
        wd = _tone_weight(yb, p.dodge_tones)
        base = (p.dodge / 100) * DODGE_EV * wd
        if skin_ok:
            d = base * skin_m
            if p.protect_face:
                d = d * (1 - 0.8 * _smoothstep(0.60, 0.85, yb))          # hold back on bright skin
            ev_skin += d
        elif person_ok:
            ev_subj += base * subj_m * 0.8
        else:
            ev_glob += base * 0.5                                         # no subject mask: stay conservative

    # ---- burn ------------------------------------------------------------------------------------
    if p.burn > 0:
        wb = _tone_weight(yb, p.burn_tones)
        base = (p.burn / 100) * BURN_EV * wb
        if bg_m is not None:
            ev_bg -= base * bg_m
        else:
            b = base * 0.5
            ev_glob -= b * (1 - face_guard) if face_guard is not None else b

    # ---- subject / background separation ----------------------------------------------------------
    if p.separation > 0 and bg_m is not None:
        ys, xs = np.nonzero(subject > 0.5)
        cy, cx = ys.mean() / h, xs.mean() / w
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.hypot((gx / w - cx), (gy / h - cy) * h / w) / 0.9
        fall = _smoothstep(0.08, 0.8, dist)                                  # 0 near the subject, 1 far away
        ev_bg -= (p.separation / 100) * SEP_BURN_EV * bg_m * (0.35 + 0.65 * fall)
        lift = subj_m if skin_m is None else np.clip(subj_m - 0.5 * skin_m, 0, 1)   # lift clothes/hair more than skin
        ev_subj += (p.separation / 100) * SEP_LIFT_EV * lift

    # ---- form: sculpt the face the way a retoucher dodges and burns --------------------------------------
    # Two cues, both measured, nothing invented:
    #   1. the light that is already on the face: brighter parts get brighter, hollows get deeper (normalised by
    #      how much light variation this face has, so it also works on a flatly lit one);
    #   2. the face's own shape: the centre (forehead, nose bridge, cheekbones) is lifted and the contour (temples,
    #      cheek sides, jaw) is deepened, the classic sculpting pattern.
    # Only broad lighting changes, never texture, and the result is confined to the face and kept soft.
    if p.form > 0 and skin_ok:
        face = mk("face")
        fm = np.clip(face, 0, 1) if (face is not None and _reliable(face, 0.002)) else skin_m
        sel = fm > 0.6
        face_px = float(np.sqrt(max(sel.sum(), 1)))
        yl = np.clip(edge_aware_blur(y, max(3.0, face_px / 5), 0.015), 0, 1)
        ref_y = float(np.median(yl[sel])) if sel.any() else 0.45
        spread = float(np.std(yl[sel])) if sel.any() else 0.1
        existing = np.clip((yl - ref_y) / max(1.4 * spread, 0.03), -1, 1)       # follows the real lighting
        dist = cv2.distanceTransform((fm > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        geo = np.clip(dist / max(float(dist.max()), 1.0), 0, 1)
        geo = geo * geo * (3 - 2 * geo)                                           # centre 1 -> contour 0
        centre = np.clip((geo - 0.45) * 2.0, -1, 1)
        # contour is deepened fully, the centre only gently lifted (a bright centre blob looks fake)
        shape = 0.5 * existing + np.where(centre < 0, 1.0 * centre, 0.4 * centre)
        shape = cv2.GaussianBlur(shape.astype(np.float32), (0, 0), max(2.0, face_px / 14))
        # the face map covers the face; the rest of the skin (neck, shoulders) follows only the real light, softly
        ev_skin += (p.form / 100) * FORM_EV * (fm * np.clip(shape, -1, 1) + (skin_m - fm).clip(0, 1) * 0.5 * existing)

    def smooth_in(ev: np.ndarray, domain: np.ndarray, sigma: float) -> np.ndarray:
        """Smooth `ev` using only pixels inside `domain` (normalised convolution), then confine it to the domain."""
        if not np.any(ev):
            return ev
        num = cv2.GaussianBlur(ev * domain, (0, 0), sigma)
        den = cv2.GaussianBlur(domain, (0, 0), sigma)
        out = num / np.maximum(den, 1e-3)
        return out * np.clip(cv2.GaussianBlur(domain, (0, 0), 1.2), 0, 1)

    ev = ev_glob.copy()
    if np.any(ev):
        ev = cv2.GaussianBlur(ev, (0, 0), soft)
    if skin_m is not None:
        ev += smooth_in(ev_skin, skin_m, soft * 0.6)
    if subj_m is not None:
        ev += smooth_in(ev_subj, subj_m, soft)
    if bg_m is not None:
        ev += smooth_in(ev_bg, bg_m, soft * 1.5)
    ev = (ev * (p.strength / 100.0)).astype(np.float32)
    if s < 1:
        ev = cv2.resize(ev, (W, H), interpolation=cv2.INTER_CUBIC)
    return ev


def apply(arr: np.ndarray, masks: Dict[str, np.ndarray], p: DodgeBurn) -> np.ndarray:
    """Apply dodge & burn. Returns `arr` itself when there is nothing to do (no copy), else a new array."""
    ev = compute_ev(arr, masks, p)
    if ev is None:
        return arr
    y = arr @ LUMA
    mx = arr.max(axis=2)                                                   # the brightest channel is what clips first
    pos = np.maximum(ev, 0) * (1 - _smoothstep(0.62, 0.96, mx))             # highlight headroom: no clipped dodging
    neg = np.minimum(ev, 0) * _smoothstep(0.03, 0.20, y)                   # shadow protection: no crushed burning
    gain = np.exp2(pos + neg)[..., None]
    lin = np.power(np.clip(arr, 0, 1), GAMMA) * gain
    lin = np.where(lin > 0.85, 0.85 + 0.15 * np.tanh((lin - 0.85) / 0.15), lin)   # soft shoulder
    return np.power(np.clip(lin, 0, 1), 1 / GAMMA).astype(np.float32)


# ---------------------------------------------------------------------------------------------------
# automatic lighting
# ---------------------------------------------------------------------------------------------------
def auto_lighting(arr: np.ndarray, masks: Dict[str, np.ndarray]) -> Tuple[DodgeBurn, List[str]]:
    """Analyse luminance and the subject and propose conservative dodge/burn values.

    Looks at: is the subject (skin, else person) under-exposed? is the background brighter than the subject or
    holding bright hot spots? is the face lit flat? With no reliable subject it only proposes gentle global
    corrections. Returns (parameters, plain-language reasons)."""
    H, W = arr.shape[:2]
    s = min(1.0, WORK_EDGE / max(H, W))
    a = _small(arr, s)
    y = (a @ LUMA).astype(np.float32)

    def mk(name):
        m = masks.get(name)
        return None if m is None else cv2.resize(m.astype(np.float32), (y.shape[1], y.shape[0]), interpolation=cv2.INTER_AREA)

    skin, person, fg, bg = mk("skin"), mk("person"), mk("foreground"), mk("background")
    subject = person if _reliable(person, 0.02) else fg
    skin_ok, person_ok = _reliable(skin, 0.002), _reliable(subject, 0.02)
    sep_cap = 55.0 if _reliable(person, 0.02) else 25.0       # a depth-derived foreground (landscape) is separated only gently
    if bg is None and person_ok:
        bg = 1.0 - subject
    res, notes = DodgeBurn(), []

    if skin_ok or person_ok:
        sel = (skin > 0.6) if skin_ok else (subject > 0.6)
        sy = float(np.median(y[sel]))
        face_share = float((skin > 0.6).mean()) if skin_ok else 0.0
        closeup = face_share > 0.03          # the face is a large part of the frame: a portrait
        target = 0.42
        under = target - sy
        if under > 0.03:
            res.dodge = float(np.clip(under * 267, 0, 50))      # about +1.6 EV per unit of shortfall, capped
            res.dodge_tones = float(np.clip(sy * 100, 20, 70))   # act on the tones the subject actually sits in
            notes.append(f"Subject is a little dark (brightness {sy:.2f}) → dodge")
        elif closeup and sy < 0.68:
            res.dodge = 24.0                                      # lift the face a touch even when exposure is fine
            res.dodge_tones = float(np.clip(sy * 100, 25, 65))
            notes.append("Close portrait → gentle dodge of the face for presence")
        elif sy > 0.74:
            notes.append("Subject is already bright → no dodge, highlights protected")
        if skin_ok:
            yl = np.clip(edge_aware_blur(y, max(3.0, float(np.sqrt((skin > 0.6).sum())) / 5), 0.015), 0, 1)
            spread = float(np.std(yl[skin > 0.6]))
            if spread < 0.05:
                res.form = 65.0 if closeup else 30.0
                notes.append("Face is lit quite flat → sculpt highlights and shadows (form)")
            elif spread < 0.09:
                res.form = 55.0 if closeup else 22.0
                notes.append("Face lighting is soft → moderate sculpting (form)")
            else:
                res.form = 42.0 if closeup else 8.0
                notes.append("Close-up face → sculpt the existing light and shadow" if closeup else "Face already has good light and shadow → only a light form touch")
        if bg is not None and (bg > 0.6).sum() > 100:
            by = float(np.median(y[bg > 0.6]))
            gap = by - sy
            if by < 0.22:
                notes.append("Dark low-key backdrop is part of the look → leave it, work on the face")
            elif gap > 0.06:
                res.separation = float(np.clip(gap * 130 + (12 if closeup else 0), 0, sep_cap))
                notes.append(f"Background is brighter than the subject by {gap:.2f} → darken it to separate the subject")
            elif closeup and by >= 0.22:
                res.separation = min(22.0, sep_cap)
                notes.append("Close portrait → slight background separation")
            hot = float(((y > 0.88) & (bg > 0.6)).sum() / max((bg > 0.6).sum(), 1))
            if hot > 0.03:
                res.burn = float(np.clip(hot * 350, 0, 35))
                res.burn_tones = 85.0
                notes.append("Bright hot spots in the background → gentle burn of those highlights")
        if not notes:
            notes.append("Lighting already well balanced → nothing to change")
    else:
        med, p98 = float(np.median(y)), float(np.percentile(y, 98))
        notes.append("No reliable subject or skin mask → only conservative global corrections")
        if med < 0.30:
            res.dodge = float(np.clip((0.36 - med) * 200, 0, 25))
            res.dodge_tones = 25.0
            notes.append("Whole picture is dark → gentle shadow lift")
        if p98 > 0.96 and float((y > 0.93).mean()) > 0.02:
            res.burn = 12.0
            res.burn_tones = 92.0
            notes.append("Large bright areas → gentle highlight burn")
    return res, notes
