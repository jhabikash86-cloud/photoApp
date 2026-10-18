"""Noise reduction: colour (chroma) noise and luminance noise, plus a measure of how noisy a photo is.

Chroma noise is blurred away with an edge-aware filter guided by the luminance, so colours do not bleed
across edges. Luminance noise uses non-local means on the luma channel only, so colour is untouched.
"""
import cv2
import numpy as np

from landscape import _guided


def estimate_sigma(rgb8: np.ndarray) -> float:
    """Noise standard deviation in 8-bit units (Immerkaer's Laplacian method). A low percentile of the residual
    is used, so real texture (grass, foliage) can't be mistaken for noise. Measured on a native-resolution crop."""
    g = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = g.shape
    c = g[h // 2 - 512:h // 2 + 512, w // 2 - 512:w // 2 + 512] if min(h, w) > 1100 else g
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = cv2.filter2D(c, -1, k)[1:-1, 1:-1]
    return float(np.percentile(np.abs(r), 35) / (0.385 * 6))


def auto_amounts(sigma: float) -> tuple:
    """(luminance, colour) noise reduction for a measured sigma. Clean files get none."""
    lum = float(np.clip((sigma - 2.2) * 10, 0, 55))
    col = float(np.clip((sigma - 0.8) * 14, 8, 60))
    return lum, col


def apply(arr: np.ndarray, lum: float, col: float) -> np.ndarray:
    """lum, col in 0..100. The correction is capped at noise-sized amounts, so strong edges and real texture
    are never smeared or bled across. Returns a new array (or the input when both are 0)."""
    if lum <= 0 and col <= 0:
        return arr
    ycc = cv2.cvtColor(np.clip(arr, 0, 1), cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0]
    edge = max(arr.shape[:2])
    if col > 0:
        c = col / 100
        r = max(3, int(edge / 200 * (1 + 3 * c)))
        cap = 0.010 + 0.040 * c                          # chroma changes beyond this are real colour, not noise
        for ch in (1, 2):
            orig = ycc[..., ch].astype(np.float32)
            smooth = _guided(y, orig, r=r, eps=4e-4 * (1 + 40 * c))
            ycc[..., ch] = orig + np.clip(smooth - orig, -cap, cap)
    if lum > 0:
        l = lum / 100
        y8 = np.clip(y * 255 + 0.5, 0, 255).astype(np.uint8)
        den = cv2.fastNlMeansDenoising(y8, None, h=2 + 10 * l, templateWindowSize=5, searchWindowSize=13)
        den = den.astype(np.float32) / 255
        cap = (2.0 + 16.0 * l) / 255                      # luminance changes beyond this are texture/detail
        ycc[..., 0] = y + np.clip(den - y, -cap, cap)
    return np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0, 1)
