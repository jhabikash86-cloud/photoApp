"""Portrait retouching driven by face landmarks.

All operations are local and masked: frequency-separation skin smoothing (even tone, keep pores),
volume sculpting (dodge & burn), eye and lip enhancement, and a soft background falloff.
Arrays are float32 RGB in [0, 1].
"""
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import scene as sc

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _fill(shape: Tuple[int, int], poly: np.ndarray, offset: np.ndarray) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull((poly - offset).astype(np.int32)), 255)
    return m.astype(np.float32) / 255.0


def _blur(m: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(m, (0, 0), max(0.5, sigma))


def _luma(a: np.ndarray) -> np.ndarray:
    return a @ LUMA


CREASE_SOFT = 0.55          # how much of a faint crease is softened (0 = off)


def retouch_face(arr: np.ndarray, face: sc.Face, strength: float, masks: Dict[str, np.ndarray],
                 smoothing: float = 1.0, face_ev: float = 0.0, smooth: Optional[float] = None,
                 eyes_amt: Optional[float] = None, tone_amt: Optional[float] = None,
                 sculpt_amt: Optional[float] = None, lips_amt: Optional[float] = None) -> None:
    """Retouch one face in place. `strength` is 0..1 and the default for every part; smooth / eyes_amt /
    tone_amt / sculpt_amt / lips_amt override it per part. `masks` accumulates full-frame masks."""
    H, W = arr.shape[:2]
    pts = face.pts
    fw = face.width
    pad = fw * 0.15
    top_pad = fw * 0.42                                             # room above the face outline: the outline cuts across the MIDDLE of the forehead, the skin goes on to the hairline
    x0, y0 = np.maximum(pts[sc.OVAL].min(0) - [pad, top_pad], 0).astype(int)
    x1, y1 = np.minimum(pts[sc.OVAL].max(0) + pad, [W, H]).astype(int)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return
    off = np.array([x0, y0], np.float32)
    crop = arr[y0:y1, x0:x1].copy()
    shape = crop.shape[:2]

    # ---- Masks -------------------------------------------------------------------------------
    oval = _fill(shape, pts[sc.OVAL], off)
    oval = cv2.erode(oval, np.ones((3, 3), np.uint8), iterations=max(1, int(fw * 0.012)))

    def feature(idx_list, grow):
        m = _fill(shape, pts[idx_list], off)
        k = max(1, int(fw * grow))
        return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))

    eyes_l = _fill(shape, pts[sc.LEFT_EYE], off)
    eyes_r = _fill(shape, pts[sc.RIGHT_EYE], off)
    eyes = np.maximum(eyes_l, eyes_r)
    lips_out = _fill(shape, pts[sc.LIPS_OUTER], off)
    lips_in = _fill(shape, pts[sc.LIPS_INNER], off)
    lips = np.clip(lips_out - lips_in, 0, 1)

    exclude = np.maximum.reduce([
        feature(sc.LEFT_EYE, 0.035), feature(sc.RIGHT_EYE, 0.035),
        feature(sc.LEFT_BROW, 0.02), feature(sc.RIGHT_BROW, 0.02),
        feature(sc.LIPS_OUTER, 0.015),
    ])
    # The face outline stops mid-forehead. Retouching only inside it leaves a visible step (a 'line') across the forehead, so the skin area is
    # extended upward to the hairline; the colour gate below keeps hair out, and the top edge is feathered over a wide band.
    ext = int(fw * 0.30)
    oval_up = cv2.dilate(oval, np.ones((ext + 1, 1), np.uint8), anchor=(0, 0)) if ext > 1 else oval            # grows the mask UPWARD only (anchor at the top row)
    oval_up = np.maximum(oval, oval_up)
    prior_core = np.clip(oval - exclude, 0, 1)
    prior = np.clip(oval_up - exclude, 0, 1)

    # Colour gate in Lab, relative to this face's own median: adapts to any skin tone and rejects
    # lipstick, beards, teeth and stray background, but keeps blemishes (small deviations).
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab)
    sel = prior_core > 0.9
    if sel.sum() < 200:
        return
    med = np.median(lab[sel], axis=0)
    dist = np.hypot(lab[..., 1] - med[1], lab[..., 2] - med[2])
    gate = np.clip((34 - dist) / 12, 0, 1)
    lum_gate = np.clip(1 - np.abs(lab[..., 0] - med[0]) / 70, 0, 1)
    skin = _blur(prior * gate * lum_gate, fw * 0.012)
    # the added forehead strip fades out toward its top edge instead of ending in a line
    rows = np.arange(shape[0], dtype=np.float32)[:, None]
    top_edge = np.where(oval_up > 0.5, 1.0, 0.0).argmax(0).astype(np.float32)             # first row of the extended area, per column
    fade = np.clip((rows - top_edge[None, :]) / max(2.0, fw * 0.16), 0, 1)
    fade = fade * fade * (3 - 2 * fade)
    skin = np.clip(skin * np.where(oval > 0.5, 1.0, fade), 0, 1)
    skin = _blur(skin, fw * 0.02)

    s = float(np.clip(strength, 0, 1))
    pick = lambda v: s if v is None else float(np.clip(v, 0, 1.5))
    s_smooth, s_eyes, s_tone, s_sculpt, s_lips = pick(smooth), pick(eyes_amt), pick(tone_amt), pick(sculpt_amt), pick(lips_amt)
    out = crop.copy()

    # ---- 1. Frequency-separation skin smoothing -----------------------------------------------
    work = min(1.0, 900.0 / fw)
    small = cv2.resize(out, None, fx=work, fy=work, interpolation=cv2.INTER_AREA) if work < 1 else out
    fws = fw * work
    base = small
    for _ in range(2):
        base = cv2.bilateralFilter(base, 0, 0.05 + 0.07 * s_smooth * smoothing, max(2.0, fws * 0.012))
    if work < 1:
        base = cv2.resize(base, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    texture = out - cv2.GaussianBlur(out, (0, 0), max(0.8, fw * 0.004))
    smoothed = base + texture * (0.62 if face.focus != "blurry" else 0.3)
    amt = (skin * min(1.0, s_smooth) * min(1.0, 0.55 + 0.45 * smoothing))[..., None]
    out = out * (1 - amt) + smoothed * amt

    # ---- 2. Sculpt volume (gentle dodge & burn) + skin glow --------------------------------------
    y = _luma(out)
    low = cv2.GaussianBlur(y, (0, 0), max(2.0, fw * 0.22))
    y_new = y + (0.30 * s_sculpt) * (y - low)
    med_y = float(np.median(y[sel]))
    lift = float(np.clip(0.42 / max(med_y, 0.05), 1.0, 1.0 + 0.15 * s_sculpt)) if med_y < 0.42 else 1.0
    y_new = y_new * (1 + (lift - 1) * skin)
    gain = (y_new + 1e-3) / (y + 1e-3)
    out = out * (1 + (gain - 1) * skin)[..., None]
    if face_ev:  # gentle, skin-only brightening
        out = out * (1 + (2 ** (face_ev * pick(None)) - 1) * skin)[..., None]

    # Skin-tone neutralisation: pull an orange/red cast back toward a natural skin hue angle
    # in Lab (natural skin sits around 45-58 degrees whatever the depth of tone).
    lab2 = cv2.cvtColor(np.clip(out, 0, 1), cv2.COLOR_RGB2Lab)
    ang = np.degrees(np.arctan2(med[2], med[1]))
    over = float(np.clip(ang - 54.0, 0, 20))
    if over > 0:
        rot = np.radians(-over * 0.8) * skin * s_tone
        a_, b_ = lab2[..., 1].copy(), lab2[..., 2].copy()
        lab2[..., 1] = a_ * np.cos(rot) - b_ * np.sin(rot)
        lab2[..., 2] = a_ * np.sin(rot) + b_ * np.cos(rot)
    lab2[..., 1:] *= (1 - 0.06 * skin * s_tone)[..., None]  # slightly less chroma: calmer, creamier skin
    out = cv2.cvtColor(lab2, cv2.COLOR_Lab2RGB)

    # ---- Crease softening, done AFTER the brightening and sculpting (they scale the contrast of every crease): they make faint forehead creases, under-eye lines and similar suddenly visible, and a pro
    # softens them by about half (never removes them: a face without lines looks fake). Only mid-scale, low-amplitude, luminance-only detail inside the skin.
    if s_smooth > 0.02 and CREASE_SOFT > 0:
        yy = _luma(out)
        g1 = cv2.GaussianBlur(yy, (0, 0), max(1.0, fw * 0.006))
        g2 = cv2.GaussianBlur(yy, (0, 0), max(3.0, fw * 0.035))
        band = g1 - g2
        rel = band / (g2 + 0.05)                                           # contrast relative to the local brightness, so brightening cannot hide the measure
        soft_band = (g2 + 0.05) * 0.06 * np.tanh(rel / 0.06)                          # only gentle detail: strong edges (brows, hairline, nostrils) are left alone
        core = _blur(cv2.erode((skin > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(1, int(fw * 0.01))).astype(np.float32), fw * 0.01)
        k_c = CREASE_SOFT * min(1.0, s_smooth * 1.4) * core
        dy = -soft_band * np.where(band < 0, 1.0, 0.45) * k_c               # dark creases are softened more than bright ridges
        out = (out * (1 + dy / (yy + 1e-3))[..., None]).astype(np.float32)

    # ---- 3. Eyes: brighter, crisper, slightly richer iris -----------------------------------------
    eye_m = _blur(eyes, fw * 0.006)[..., None]
    y = _luma(out)[..., None]
    e = out * (1 + 0.06 * s_eyes)
    e = y * 1.0 + (e - y) * 1.0
    mean = float(_luma(out)[eyes > 0.5].mean()) if (eyes > 0.5).any() else 0.4
    e = (e - mean) * (1 + 0.08 * s_eyes) + mean
    sharp = e + (e - cv2.GaussianBlur(e, (0, 0), max(0.8, fw * 0.006))) * (0.5 * s_eyes)
    out = out * (1 - eye_m) + sharp * eye_m

    # ---- 4. Lips: richer colour and depth, no invented makeup -------------------------------------
    lip_m = _blur(lips, fw * 0.005)[..., None]
    g = _luma(out)[..., None]
    lipc = g + (out - g) * (1 + 0.28 * s_lips)
    lipc = lipc * (1 - 0.06 * s_lips)
    out = out * (1 - lip_m) + lipc * lip_m

    arr[y0:y1, x0:x1] = np.clip(out, 0, 1)

    # ---- Bookkeeping for protection / debug overlay -----------------------------------------------
    for name, m in (("skin", skin), ("eyes", eyes), ("lips", lips)):
        full = masks.setdefault(name, np.zeros((H, W), np.float32))
        region = full[y0:y1, x0:x1]
        np.maximum(region, m, out=region)


def background_falloff(arr: np.ndarray, face: sc.Face, strength: float) -> None:
    """Darken edges away from the face so the subject separates from the background (in place)."""
    H, W = arr.shape[:2]
    cx, cy = face.pts[sc.OVAL].mean(0)
    fh = float(np.ptp(face.pts[sc.OVAL][:, 1]))
    R = min(fh * 2.8, 0.42 * float(np.hypot(W, H)))
    sub = max(1, int(max(H, W) / 800))
    ys, xs = np.mgrid[0:H:sub, 0:W:sub].astype(np.float32)
    r = np.hypot((xs - cx) / 1.25, ys - cy) / R
    d = np.clip((r - 0.30) / 1.1, 0, 1)
    d = d * d * (3 - 2 * d)
    gain = 1 - 0.20 * strength * d
    gain = cv2.resize(gain, (W, H), interpolation=cv2.INTER_LINEAR)
    arr *= gain[..., None]


def dreamy_glow(arr: np.ndarray, amount: float, sky: np.ndarray = None, skin: np.ndarray = None) -> np.ndarray:
    """Soft highlight bloom (pro-mist style) with a faint warm tint and slightly lifted blacks."""
    long_edge = max(arr.shape[:2])
    hi = np.clip(arr - 0.45, 0, 1) / 0.55
    bloom = cv2.GaussianBlur(hi, (0, 0), long_edge / 90)
    out = 1 - (1 - arr) * (1 - bloom * amount)               # screen blend
    warm = amount * 0.06 * np.array([1.0, 0.35, -0.7], np.float32)
    keep = np.zeros(arr.shape[:2], np.float32)
    for m in (sky, skin):   # no warm tint on the sky, and none on skin (it would only make people orange)
        if m is not None:
            keep = np.maximum(keep, np.clip(m, 0, 1))
    w = (1 - keep[..., None])
    out = out * (1 + warm * w)  # a whisper of warmth
    return np.clip(out * (1 - 0.012) + 0.012, 0, 1)             # lift blacks a touch


def skin_reference_angle(arr: np.ndarray, face: sc.Face) -> float:
    """Median Lab hue angle (degrees) of the face's cheeks/forehead, used to tone-match neck and body."""
    H, W = arr.shape[:2]
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(face.pts[sc.OVAL].astype(np.int32)), 255)
    for idx in (sc.LEFT_EYE, sc.RIGHT_EYE, sc.LEFT_BROW, sc.RIGHT_BROW, sc.LIPS_OUTER):
        cv2.fillConvexPoly(m, cv2.convexHull(face.pts[idx].astype(np.int32)), 0)
    ys, xs = np.nonzero(m)
    if len(ys) < 50:
        return 52.0
    step = max(1, len(ys) // 4000)
    lab = cv2.cvtColor(arr[ys[::step], xs[::step]].reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)
    med = np.median(lab, axis=0)
    return float(np.degrees(np.arctan2(med[2], med[1])))


def tone_body_skin(arr: np.ndarray, mask: np.ndarray, strength: float, ev: float) -> None:
    """Neck/arms/legs: gentle brightening and the same orange-cast correction as the face (in place)."""
    if mask.max() < 0.05:
        return
    s = float(np.clip(strength, 0, 1))
    m = mask[..., None]
    arr[:] = np.clip(arr * (1 + (2 ** (ev * s) - 1) * m), 0, 1)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2Lab)
    sel = mask > 0.5
    if sel.sum() < 100:
        return
    med = np.median(lab[sel], axis=0)
    ang = np.degrees(np.arctan2(med[2], med[1]))
    over = float(np.clip(ang - 54.0, 0, 20))
    if over > 0:
        rot = np.radians(-over * 0.8) * mask * s
        a_, b_ = lab[..., 1].copy(), lab[..., 2].copy()
        lab[..., 1] = a_ * np.cos(rot) - b_ * np.sin(rot)
        lab[..., 2] = a_ * np.sin(rot) + b_ * np.cos(rot)
    lab[..., 1:] *= (1 - 0.05 * mask * s)[..., None]
    arr[:] = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
