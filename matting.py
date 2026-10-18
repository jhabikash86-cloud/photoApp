"""Alpha matting for the subject: clean edges at the *render* resolution, with colour decontamination.

The segmentation model gives a soft person mask at working size. That is fine for tone work, but for a background
blur or a cut-out look every defect shows: speckle, halos, and background colour baked into hair and edge pixels.
This module turns the model mask into a proper matte:

  1. clean       remove speckle and pinholes (small components), smooth the mask boundary;
  2. trimap      sure foreground, sure background, and a narrow unknown band along the boundary;
  3. alpha       inside the unknown band the matte follows the *picture's* real edges (guided filter at render
                 resolution). Hair gets a fine, wispy matte; clothes, skin and objects get a crisp, clean edge;
  4. decontaminate  edge pixels are re-coloured from the nearby solid subject, so no background colour (green
                 grass, blue sky) is carried into the new composite.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from landscape import _guided


def _odd(n: int) -> int:
    return max(3, int(n) | 1)


def clean_mask(m: np.ndarray, edge: int, hair: Optional[np.ndarray] = None) -> np.ndarray:
    """Remove speckle (tiny islands) and fill pinholes, keeping the model's soft edge values. m float32 0..1."""
    min_area = max(30, int(0.00035 * m.size))
    b = (m > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    keep = np.zeros_like(b)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    holes = np.zeros_like(b)
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - keep)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_area * 2 and x > 0 and y > 0 and x + w < m.shape[1] and y + h < m.shape[0]:
            holes[lab == i] = 1
    k = _odd(edge / 500)
    near = cv2.dilate(keep, np.ones((k, k), np.uint8)).astype(np.float32)
    # hysteresis: faint pixels (loose hair, fur, soft edges) are real only when they connect to the solid subject;
    # isolated faint islands are speckle. This keeps strands and removes noise.
    low = (m > 0.10).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
    touching = np.unique(lab[keep > 0])
    linked = np.isin(lab, touching[touching > 0]).astype(np.float32)
    near = np.maximum(near, linked)
    out = m * near
    return np.where(holes > 0, np.maximum(out, 0.95), out).astype(np.float32)


def refine_alpha(img: np.ndarray, m: np.ndarray, hair: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (alpha, sure_foreground). img float32 RGB 0..1, m person mask float32 0..1 at the same size."""
    H, W = m.shape
    edge = max(H, W)
    mc = clean_mask(m, edge, hair)
    k = _odd(edge / 260)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sure_fg = cv2.erode((mc > 0.85).astype(np.uint8), kern).astype(np.float32)
    sure_bg = 1 - cv2.dilate((mc > 0.08).astype(np.uint8), kern).astype(np.float32)
    unknown = np.clip(1 - sure_fg - sure_bg, 0, 1)

    gray = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_RGB2GRAY).astype(np.float32)
    s = min(1.0, 1400 / edge)
    gs = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    ms = cv2.resize(cv2.GaussianBlur(mc, (0, 0), max(0.8, edge / 900)), (gs.shape[1], gs.shape[0]),
                    interpolation=cv2.INTER_AREA) if s < 1 else cv2.GaussianBlur(mc, (0, 0), max(0.8, edge / 900))
    r = max(2, int(max(gs.shape) / 330))
    fine = np.clip(_guided(gs, ms, r=r, eps=2e-5), 0, 1)          # follows strands and fine edges
    crisp = np.clip(_guided(gs, ms, r=r + 1, eps=6e-4), 0, 1)      # follows hard edges, ignores texture
    if s < 1:
        fine = cv2.resize(fine, (W, H), interpolation=cv2.INTER_CUBIC)
        crisp = cv2.resize(crisp, (W, H), interpolation=cv2.INTER_CUBIC)
    crisp = np.clip((crisp - 0.30) / 0.40, 0, 1)
    crisp = crisp * crisp * (3 - 2 * crisp)                        # hard, clean contour for non-hair edges

    if hair is not None:
        hk = _odd(edge / 90)
        hw = cv2.GaussianBlur(cv2.dilate(np.clip(hair, 0, 1), np.ones((hk, hk), np.uint8)), (0, 0), edge / 300)
        hw = np.clip(hw * 1.4, 0, 1)
    else:
        hw = np.zeros_like(m)
    band_alpha = hw * np.clip(fine, 0, 1) + (1 - hw) * crisp
    alpha = sure_fg + unknown * band_alpha
    alpha = np.clip(alpha, 0, 1) * (1 - sure_bg)
    return alpha.astype(np.float32), sure_fg


def decontaminate(img: np.ndarray, alpha: np.ndarray, sure_fg: np.ndarray) -> np.ndarray:
    """Foreground colours with the old background's colour removed from the edge pixels."""
    H, W = alpha.shape
    edge = max(H, W)
    s = min(1.0, 900 / edge)
    sm = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    w = cv2.resize(sure_fg, (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_AREA) if s < 1 else sure_fg
    sig = max(3.0, max(sm.shape) / 130)
    num = cv2.GaussianBlur(sm * w[..., None], (0, 0), sig)
    den = cv2.GaussianBlur(w, (0, 0), sig)[..., None]
    fhat = num / np.maximum(den, 1e-3)
    fhat = np.where(den > 0.02, fhat, sm)
    if s < 1:
        fhat = cv2.resize(fhat, (W, H), interpolation=cv2.INTER_CUBIC)
    wf = np.clip((0.985 - alpha) / 0.55, 0, 1) ** 0.7
    return np.clip(img * (1 - wf[..., None]) + fhat * wf[..., None], 0, 1).astype(np.float32)
