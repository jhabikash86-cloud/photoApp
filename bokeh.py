"""Portrait background blur ("bokeh") that behaves like a real lens.

- The subject plane stays sharp. Blur grows smoothly with distance from the subject (a depth proxy),
  so the area right behind the subject is only slightly soft and distant scenery is melted away.
- Blur uses a round disc kernel (lens-like), not a Gaussian, and is computed with normalised convolution
  over the background only, so the subject's colours never bleed into the blur (no halo around hair).
- Bright points of light are emphasised before blurring, so they become soft discs.
- If a depth map is available it replaces the distance proxy.
"""
from typing import Optional

import matting

import cv2
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _disc(r: float) -> np.ndarray:
    k = max(1, int(round(r)))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    d = np.hypot(xx, yy)
    ker = np.clip(k + 0.5 - d, 0, 1).astype(np.float32)   # antialiased edge
    return ker / ker.sum()


def _disc_blur(img: np.ndarray, r: float) -> np.ndarray:
    """Disc blur of radius r px; large radii are computed at reduced size for speed."""
    if r < 0.8:
        return img
    f = max(1.0, r / 7.0)
    if f > 1:
        small = cv2.resize(img, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
        out = cv2.filter2D(small, -1, _disc(r / f), borderType=cv2.BORDER_REFLECT)
        return cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.filter2D(img, -1, _disc(r), borderType=cv2.BORDER_REFLECT)


def blur_map_from_subject(subject: np.ndarray, far: Optional[np.ndarray] = None) -> np.ndarray:
    """0 = in focus (the subject), 1 = maximum blur. Distance from the subject stands in for depth."""
    H, W = subject.shape
    inside = (subject > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
    ref = 0.16 * max(H, W)
    b = np.clip(dist / ref, 0, 1) ** 0.65
    if far is not None:   # sky and distant mountains are at "infinity": fully blurred
        b = np.maximum(b, np.clip(far, 0, 1) * 0.95)
    b = cv2.GaussianBlur(b.astype(np.float32), (0, 0), max(2.0, ref / 12))   # no abrupt steps
    return np.clip(b * (1 - np.clip(subject, 0, 1)), 0, 1)


def blur_map_from_depth(depth: np.ndarray, subject: np.ndarray, focus: float = 50) -> np.ndarray:
    """0 = in focus, 1 = maximum blur. Blur grows with the depth difference from the subject, both behind it
    and in front of it (like a real lens). `focus` (0..100) widens the in-focus zone around the subject."""
    H, W = subject.shape
    inside = subject > 0.5
    d_subject = float(np.median(depth[inside])) if inside.any() else 0.5
    zone = 0.03 + 0.22 * focus / 100                      # how much depth counts as "still sharp"
    diff = np.abs(depth - d_subject)
    b = np.clip((diff - zone) / 0.45, 0, 1) ** 0.8
    # the matte (not a sharp margin) handles the transition at the outline; a margin would leave a crisp halo of
    # background around the subject. Only a hair-thin ramp stops depth noise right at the contour.
    dist = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 5)
    b = b * np.clip(dist / (0.002 * max(H, W) + 1), 0, 1)
    b = cv2.GaussianBlur(b.astype(np.float32), (0, 0), max(1.5, max(H, W) / 500))
    return np.clip(b * (1 - np.clip(subject, 0, 1)), 0, 1)


def apply(arr: np.ndarray, subject: np.ndarray, amount: float, size: float = 50, highlights: float = 40,
          far: Optional[np.ndarray] = None, depth: Optional[np.ndarray] = None, focus: float = 50,
          hair: Optional[np.ndarray] = None, noise: float = 0.0, matte=None) -> np.ndarray:
    """arr float32 RGB [0,1]; subject soft mask (HxW); amount/size/highlights/focus 0..100.
    `matte` = precomputed (alpha, sure_fg) from matting.refine_alpha; `noise` = source noise sigma (8-bit units),
    used to put matching grain back into the blurred area."""
    if amount <= 0 or subject is None or (subject > 0.5).mean() < 0.01:
        return arr
    H, W = arr.shape[:2]
    edge = max(H, W)
    a = amount / 100
    radius_max = edge * (0.004 + 0.036 * a) * (0.6 + 0.8 * size / 100)
    if radius_max < 1.2:
        return arr

    alpha, sure_fg = matte if matte is not None else matting.refine_alpha(arr, subject, hair)
    b = blur_map_from_depth(depth, alpha, focus) if depth is not None else blur_map_from_subject(alpha, far)
    b = np.clip(b * (0.35 + 0.65 * min(1.0, a * 1.2)), 0, 1)

    # Background-only normalised convolution: weights are 0 on the subject (grown a little to catch hair), so the
    # subject's colours never bleed into the blur.
    grow = max(3, int(edge * 0.003)) | 1
    bgw = np.clip(1 - cv2.dilate(sure_fg, np.ones((grow, grow), np.uint8)), 0, 1).astype(np.float32)

    # Light points become soft discs: brighten only the luminance of very bright pixels (hue is untouched).
    y = arr @ LUMA
    gain = 1 + (highlights / 100) * 2.2 * np.clip((y - 0.82) / 0.18, 0, 1) ** 2
    src = arr * gain[..., None]

    radii = radius_max * np.array([0.12, 0.28, 0.55, 1.0], np.float32)
    levels = [arr]
    for r in radii:
        num = _disc_blur(src * bgw[..., None], r)
        den = _disc_blur(bgw, r)[..., None]
        blurred = num / np.maximum(den, 1e-3)
        # where almost no background is within reach, borrow from a much wider background-only blur; never from a
        # plain blur, which would drag the subject's bright parts (a white strap, a shirt) out into a glow
        r2 = r * 3.0
        wide = _disc_blur(src * bgw[..., None], r2) / np.maximum(_disc_blur(bgw, r2)[..., None], 1e-3)
        conf = np.clip(den / 0.04, 0, 1)
        levels.append(blurred * conf + wide * (1 - conf))

    pos = b * (len(levels) - 1)
    out = np.zeros_like(arr)
    for k, L in enumerate(levels):
        w = np.clip(1 - np.abs(pos - k), 0, 1)[..., None]
        out += w * L
    # soft shoulder on the LUMINANCE (scales all channels together, so colours do not drift toward yellow)
    yo = out @ LUMA
    yo_s = np.where(yo > 0.9, 0.9 + 0.1 * np.tanh((yo - 0.9) / 0.1), yo)
    out = out * (yo_s / np.maximum(yo, 1e-4))[..., None]

    # a blurred area has lost the sensor grain; put matching grain back so it does not look pasted-in
    if noise > 0.4:
        rng = np.random.default_rng(11)
        g = rng.normal(0, 1, (H, W)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), max(0.5, edge / 3500))
        wblur = np.clip(b * 1.5, 0, 1)[..., None]
        out = out + (g[..., None] * (noise / 255.0) * 0.8) * wblur

    fg = matting.decontaminate(arr, alpha, sure_fg)
    al = alpha[..., None]
    return np.clip(fg * al + out * (1 - al), 0, 1)
