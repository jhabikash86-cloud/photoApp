"""Digital long exposure: the look of a slow shutter and a neutral-density filter, made from a single frame.

A long exposure averages what moves. Water turns to glass or mist and its ripples vanish while reflections, rocks and shoreline stay
sharp; clouds stretch into streaks along the wind; a graduated ND filter holds the sky down without touching the land. Here each is done
only inside its own region (the segmentation masks), in linear light (an exposure averages light, not gamma-encoded numbers), with a
directional blur whose direction is read from the picture (flow of a stream or waterfall, otherwise horizontal) and whose length
is a fraction of the picture size, so preview and export look the same.

Everything runs at a fixed working size (the result is smooth by design) and is carried back up band by band, so a 48 MP export needs little extra memory.
"""
import math
from typing import Dict, Optional

import cv2
import numpy as np

from bands import banded
from params import LongExposure

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
WORK = 1400                       # edge of the working size: the blurs are long, so nothing is lost by doing them here


def _lin(a):
    return np.power(np.clip(a, 0, 1), 2.2).astype(np.float32)


def _srgb(a):
    return np.power(np.clip(a, 0, 1), 1 / 2.2).astype(np.float32)


class _Up:
    """A small 2-D map that hands out full-size row bands on request (bilinear), so a full-size copy is never held."""

    def __init__(self, a: np.ndarray, h: int, w: int):
        self.a, self.h, self.w = a, h, w
        self.sh = a.shape[0]

    def band(self, y0: int, y1: int) -> np.ndarray:
        r0 = max(0, int(y0 / self.h * self.sh) - 1)
        r1 = min(self.sh, int(math.ceil(y1 / self.h * self.sh)) + 2)
        hh = max(1, int(round((r1 - r0) * self.h / self.sh)))
        up = cv2.resize(self.a[r0:r1], (self.w, hh), interpolation=cv2.INTER_LINEAR)
        off = int(round(r0 * self.h / self.sh))
        sl = up[max(0, y0 - off):max(0, y1 - off)]
        if sl.shape[0] < y1 - y0:
            sl = np.concatenate([sl, np.repeat(sl[-1:] if len(sl) else up[-1:], (y1 - y0) - len(sl), axis=0)], axis=0)
        return sl


def _up(a: np.ndarray, h: int, w: int):
    if h * w >= 3_000_000:
        return _Up(a, h, w)
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR) if a.shape[:2] != (h, w) else a


@banded
def _mix_px(arr: np.ndarray, w, smooth) -> np.ndarray:
    """arr + w * (smooth - arr): the exposure-averaged picture replaces the original where w says so (smooth is 3-channel, w 1-channel)."""
    return np.clip(arr + w[..., None] * (smooth - arr), 0, 1).astype(np.float32)


@banded
def _gain_px(arr: np.ndarray, ev) -> np.ndarray:
    """A graduated ND filter in linear light: 2^-ev, ev a smooth map."""
    return _srgb(_lin(arr) * np.exp2(-ev)[..., None])


# ------------------------------------------------------------------------------------------------ direction
def streak_direction(y: np.ndarray, m: np.ndarray):
    """(angle in radians of the dominant streak direction in the picture's own coordinates, coherence 0..1) of the texture inside mask m.
    Ripples on a lake have no preferred direction (low coherence); a stream or waterfall has a clear one."""
    hp = y - cv2.GaussianBlur(y, (0, 0), max(2.0, max(y.shape) / 90))
    gx = cv2.Sobel(hp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hp, cv2.CV_32F, 0, 1, ksize=3)
    wgt = (m > 0.5).astype(np.float32)
    jxx, jyy, jxy = float((gx * gx * wgt).sum()), float((gy * gy * wgt).sum()), float((gx * gy * wgt).sum())
    tot = jxx + jyy + 1e-9
    coh = math.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / tot
    grad_angle = 0.5 * math.atan2(2 * jxy, jxx - jyy)          # direction of the steepest change
    return grad_angle + math.pi / 2, float(coh)                 # streaks run across it


def _aniso_smooth(img: np.ndarray, m: np.ndarray, sig_major: float, sig_minor: float, theta: float) -> np.ndarray:
    """Directional Gaussian blur of `img` in linear light that only averages pixels inside mask `m` (so rocks, shore and sky never leak in).
    theta: direction of the long axis in image coordinates (radians, y down)."""
    h, w = m.shape
    diag = int(math.ceil(math.hypot(h, w))) + 4
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, math.degrees(theta), 1.0)          # levels a line at angle theta
    M[0, 2] += diag / 2.0 - c[0]
    M[1, 2] += diag / 2.0 - c[1]
    hard = (m > 0.5).astype(np.float32)
    stack = np.concatenate([img * hard[..., None], hard[..., None]], axis=2)                 # (num..., den)
    rot = cv2.warpAffine(stack, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    blur = cv2.GaussianBlur(rot, (0, 0), sigmaX=max(0.5, sig_major), sigmaY=max(0.5, sig_minor))
    Mi = cv2.invertAffineTransform(M)
    back = cv2.warpAffine(blur, Mi, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    den = back[..., 3:4]
    return np.where(den > 1e-3, back[..., :3] / np.maximum(den, 1e-3), img).astype(np.float32)


def _grown(m: np.ndarray, r: int) -> np.ndarray:
    g = cv2.dilate((m > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(np.float32)
    return np.maximum(m, cv2.GaussianBlur(g, (0, 0), r / 2.0))


def _horizon(sky: np.ndarray) -> Optional[float]:
    if float((sky > 0.5).mean()) < 0.02:
        return None
    rows = np.nonzero(sky > 0.5)[0]
    return float(np.percentile(rows, 98)) / sky.shape[0]


# ------------------------------------------------------------------------------------------------ the stage
def active(le: LongExposure) -> bool:
    return bool(le.enabled and (le.water > 0 or le.clouds > 0 or le.grad_nd > 0))


def apply(src: np.ndarray, masks: Dict[str, np.ndarray], le: LongExposure) -> np.ndarray:
    """The picture with the long-exposure look. Returns `src` itself when nothing applies."""
    if not active(le):
        return src
    H, W = src.shape[:2]
    s = min(1.0, WORK / max(H, W))
    sw, sh = max(8, int(W * s)), max(8, int(H * s))
    small = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA) if s < 1 else src
    lin = _lin(small)
    ymask = lambda name: (cv2.resize(np.asarray(masks[name], np.float32), (sw, sh), interpolation=cv2.INTER_AREA) if name in masks else None)
    big = max(sh, sw)
    smooth = lin.copy()
    wmap = np.zeros((sh, sw), np.float32)
    did = False

    water = ymask("water") if le.water > 0 else None
    if water is not None and (water > 0.5).mean() > 0.01:
        a = 0.8 * float(np.clip(le.water / 100.0, 0, 1)) ** 0.8          # 100% = a strong long exposure, not a smear
        y = (small @ LUMA).astype(np.float32)
        ang, coh = streak_direction(y, water)
        mode = le.water_dir
        if mode == "horizontal":
            theta, flow = 0.0, False
        elif mode == "vertical":
            theta, flow = math.pi / 2, True
        else:
            flow = coh > 0.22                                           # a stream, a waterfall, surf: follow its flow; a lake or sea: glass
            theta = ang if flow else 0.0
        major = (0.008 + 0.034 * a) * big                              # 100% is already a strong long exposure; beyond it rocks in the mask smear
        minor = major * (0.10 if flow else 0.40)
        wsm = _aniso_smooth(lin, water, major, minor, theta)
        feather = cv2.GaussianBlur(water, (0, 0), max(1.0, 0.003 * big))
        wt = np.clip(feather * (0.40 + 0.60 * a), 0, 1)
        smooth = smooth * (1 - wt[..., None]) + wsm * wt[..., None]
        wmap = np.maximum(wmap, wt)
        did = True

    sky = ymask("sky") if (le.clouds > 0 or le.grad_nd > 0) else None
    if sky is not None and (sky > 0.5).mean() > 0.01:
        r = max(2, int(round(0.012 * big)))
        sky_g = _grown(sky, r)
        if le.clouds > 0:
            a = 0.9 * float(np.clip(le.clouds / 100.0, 0, 1)) ** 0.85
            theta = math.radians(le.cloud_angle)
            major = (0.010 + 0.045 * a) * big
            csm = _aniso_smooth(lin, sky, major, major * 0.12, theta)
            wt = np.clip(cv2.GaussianBlur(sky, (0, 0), max(1.0, 0.003 * big)) * (0.30 + 0.60 * a), 0, 1)
            smooth = smooth * (1 - wt[..., None]) + csm * wt[..., None]
            wmap = np.maximum(wmap, wt)
            did = True

    ev_map = None
    if sky is not None and le.grad_nd > 0 and (sky > 0.5).mean() > 0.01:
        hz = _horizon(sky)
        if hz is not None and hz > 0.08:
            yy = np.arange(sh, dtype=np.float32)[:, None] / (hz * sh)
            ramp = np.clip(1.0 - yy, 0, 1) ** 0.85
            ev_map = (1.8 * np.clip(le.grad_nd / 100.0, 0, 1) * ramp * _grown(sky, max(2, int(round(0.012 * big))))).astype(np.float32)

    if not did and ev_map is None:
        return src
    out = src
    if did:
        out = _mix_px(out, _up(np.clip(wmap, 0, 1), H, W), _up(_srgb(smooth), H, W))
    if ev_map is not None:
        out = _gain_px(out, _up(ev_map, H, W))
    return out
