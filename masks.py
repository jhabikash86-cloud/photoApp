"""Mask quality scoring, automatic refinement/retry, and extra region masks (foliage).

A mask is only as good as its edges. Every region mask is scored on how well its boundary follows
real image edges, how coherent it is (no confetti, no holes) and simple scene priors. If the score is
below the bar, the engine tries other refinements (bigger guided filter, GrabCut, intersections) and
keeps the best. Masks that still fail are flagged, and the engine softens edits for that region.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from landscape import _guided

PASS_SCORE = 0.72      # a mask at or above this is treated as accurate
MIN_STRENGTH = 0.35    # edits on a failing mask are scaled between this and 1.0 by confidence


@dataclass
class MaskReport:
    kind: str
    score: float = 0.0
    edge: float = 0.0
    coherence: float = 0.0
    prior: float = 1.0
    tries: List[str] = field(default_factory=list)
    chosen: str = "base"
    passed: bool = False
    note: str = ""
    floor: float = MIN_STRENGTH

    @property
    def strength(self) -> float:
        """Scale applied to this region's edits: 1.0 when accurate, softened when not."""
        if self.passed:
            return 1.0
        return float(np.clip(self.floor + (1 - self.floor) * self.score / PASS_SCORE, self.floor, 1.0))

    def summary(self) -> dict:
        return {"score": round(self.score, 2), "edge": round(self.edge, 2), "coherence": round(self.coherence, 2),
                "passed": self.passed, "chosen": self.chosen, "tries": self.tries, "note": self.note}


def _thumb(a: np.ndarray, edge: int = 512) -> np.ndarray:
    s = min(1.0, edge / max(a.shape[:2]))
    return cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else a


def score_mask(mask: np.ndarray, rgb: np.ndarray, kind: str, faces=None) -> MaskReport:
    """Score 0..1. High = boundary follows real edges, region is coherent, scene priors hold."""
    rep = MaskReport(kind=kind)
    m = _thumb(mask.astype(np.float32))
    img = _thumb(rgb)
    h, w = m.shape
    area = float((m > 0.5).mean())
    if area < 0.005:
        rep.note = "empty"
        return rep
    if area > 0.995:
        rep.note = "covers everything"
        rep.score = 0.0
        return rep

    # 1. Edge alignment: boundary pixels should sit on strong colour edges.
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab = cv2.GaussianBlur(lab, (0, 0), 1.0)
    grad = np.zeros((h, w), np.float32)
    for c, wgt in ((0, 0.6), (1, 1.0), (2, 1.0)):
        grad += wgt * np.hypot(cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0), cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1))
    binm = (m > 0.5).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    band = (cv2.dilate(binm, k) - cv2.erode(binm, k)).astype(bool)
    band[:3, :] = band[-3:, :] = False   # the picture frame is not a mask boundary
    band[:, :3] = band[:, -3:] = False
    if band.sum() < 20:
        rep.edge = 0.8  # region touches only the frame
    else:
        # Compare with the best nearby gradient (edges within +-4px) so a mask that is 1-2px off still scores.
        local_max = cv2.dilate(grad, np.ones((9, 9), np.uint8))
        ratio = float(grad[band].mean() / (grad.mean() + 1e-4))
        near = float(local_max[band].mean() / (grad.mean() + 1e-4))
        rep.edge = float(np.clip((0.5 * ratio + 0.5 * near / 1.8 - 0.9) / 1.6, 0, 1))

    # 2. Coherence: few big blobs, few holes, little confetti.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binm)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
    big = areas[areas > 0.003 * h * w].sum()
    frag = 1 - big / max(1, areas.sum())
    # Holes = background pockets fully enclosed by the mask (background components not touching the frame).
    nb, lab_bg, st_bg, _ = cv2.connectedComponentsWithStats((1 - binm).astype(np.uint8))
    holes_area = 0
    for i in range(1, nb):
        x, y, bw, bh, a_ = st_bg[i]
        if x > 0 and y > 0 and x + bw < w and y + bh < h:
            holes_area += a_
    holes = holes_area / max(1, binm.sum())
    unc = float(((m > 0.15) & (m < 0.85)).sum()) / max(1, band.sum())
    rep.coherence = float(np.clip(1 - 1.5 * frag - 2.0 * holes - 0.05 * max(0, unc - 2), 0, 1))

    # 3. Scene priors.
    prior = 1.0
    if kind == "sky":
        top = (binm[: max(2, h // 20)] > 0).mean()
        if top < 0.25:
            prior *= 0.5
            rep.note = "sky does not reach the top of the frame"
        low = float(binm[int(h * 0.6):].sum()) / max(1, binm.sum())
        if low > 0.12:  # sky does not hang in the bottom 40% of a frame: probably sea, a wall or a window
            prior *= max(0.4, 1 - 2.5 * (low - 0.12))
            rep.note = rep.note or "part of the 'sky' sits in the lower frame (sea/wall?)"
    if kind == "person" and faces:
        inside = 0
        for f in faces:
            cx, cy = (f.pts[:, 0].mean() / mask.shape[1]) * w, (f.pts[:, 1].mean() / mask.shape[0]) * h
            inside += int(m[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))] > 0.5)
        if inside < len(faces):
            prior *= 0.4 + 0.6 * inside / len(faces)
            rep.note = "a detected face is outside the person mask"
    rep.prior = prior
    rep.score = float((0.55 * rep.edge + 0.45 * rep.coherence) * prior)
    rep.passed = rep.score >= PASS_SCORE
    return rep


def grabcut_refine(rgb: np.ndarray, mask: np.ndarray, edge: int = 640, iters: int = 4) -> np.ndarray:
    """Snap a mask to colour edges: GrabCut seeded by the confident core and confident background."""
    H, W = mask.shape
    s = min(1.0, edge / max(H, W))
    small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
    m = cv2.resize(mask.astype(np.float32), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    k = np.ones((5, 5), np.uint8)
    sure_fg = cv2.erode((m > 0.9).astype(np.uint8), k, iterations=2) > 0
    sure_bg = cv2.erode((m < 0.1).astype(np.uint8), k, iterations=2) > 0
    if sure_fg.sum() < 50 or sure_bg.sum() < 50:
        return mask
    gc = np.full(m.shape, cv2.GC_PR_BGD, np.uint8)
    gc[m > 0.5] = cv2.GC_PR_FGD
    gc[sure_fg] = cv2.GC_FGD
    gc[sure_bg] = cv2.GC_BGD
    bgr = cv2.cvtColor((np.clip(small, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(bgr, gc, None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    out = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.float32)
    out = cv2.GaussianBlur(out, (0, 0), 1.2)
    guide = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = np.clip(_guided(guide, out, r=max(3, max(out.shape) // 120), eps=1e-3), 0, 1)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def guided_variant(rgb: np.ndarray, mask: np.ndarray, radius_div: int = 60, eps: float = 4e-4) -> np.ndarray:
    H, W = mask.shape
    s = min(1.0, 800 / max(H, W))
    small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
    m = cv2.resize(mask.astype(np.float32), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    guide = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = np.clip(_guided(guide, m, r=max(3, max(m.shape) // radius_div), eps=eps), 0, 1)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def best_mask(kind: str, rgb: np.ndarray, base: np.ndarray, faces=None, max_tries: int = 5,
              floor: float = MIN_STRENGTH) -> tuple:
    """Score the base mask; if it fails, try refinements until one passes (or the best is kept)."""
    base = np.clip(base.astype(np.float32), 0, 1)
    best_m, best = base, score_mask(base, rgb, kind, faces)
    best.tries = [f"base={best.score:.2f}"]
    best.floor = floor
    if best.passed or best.note in ("empty", "covers everything"):
        return best_m, best

    def sharpen(m, c=0.5, g=6.0):  # contrast the soft edge
        return np.clip((m - c) * g + 0.5, 0, 1)

    ladder: List[tuple] = [
        ("guided-tight", lambda: guided_variant(rgb, base, 90, 2e-4)),
        ("guided-wide", lambda: guided_variant(rgb, base, 40, 1e-3)),
        ("grabcut", lambda: grabcut_refine(rgb, base)),
        ("grabcut+guided", lambda: guided_variant(rgb, grabcut_refine(rgb, base), 80, 3e-4)),
        ("grabcut-limited", lambda: np.minimum(grabcut_refine(rgb, base, 800, 5),
                                               cv2.dilate(sharpen(base), np.ones((25, 25), np.uint8)))),
    ]
    for name, fn in ladder[:max_tries]:
        try:
            cand = np.clip(fn(), 0, 1)
        except Exception as e:  # a failed refinement must never break the edit
            best.tries.append(f"{name}=error")
            continue
        rep = score_mask(cand, rgb, kind, faces)
        best.tries.append(f"{name}={rep.score:.2f}")
        if rep.score > best.score:
            tries = best.tries
            best_m, best = cand, rep
            best.tries, best.chosen, best.floor = tries, name, floor
        if best.passed:
            break
    return best_m, best


def foliage_mask(rgb: np.ndarray, exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """Trees, bushes and grass: green-ish hues, away from sky and people."""
    H, W = rgb.shape[:2]
    s = min(1.0, 800 / max(H, W))
    t = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
    hsv = cv2.cvtColor(t, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (hue >= 52) & (hue <= 170) & (sat > 0.16) & (val > 0.06) & (val < 0.92)
    if exclude is not None:
        green &= cv2.resize(exclude.astype(np.float32), (t.shape[1], t.shape[0])) < 0.3
    m = green.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 0.0015 * m.size:
            keep[labels == i] = 1
    soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.5)
    guide = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY).astype(np.float32)
    soft = np.clip(_guided(guide, soft, r=max(3, max(soft.shape) // 150), eps=2e-3), 0, 1)
    return cv2.resize(soft, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1)
