"""Selective masking engine.

A MaskBank wraps the semantic masks found once at analysis time (sky, mountains, trees, grass, water, buildings,
ground, person, skin, hair) and turns them into masks that any adjustment can use:

  * every mask is a float32 array in 0..1, resized (never thresholded) to the render size, so it always aligns
    with the picture;
  * derived masks: `foreground` (the subject: person, else the near part of the depth map) and `background`;
  * set operations: invert, union, intersect, subtract (soft, so edges stay smooth);
  * `effective()` applies the user's controls: invert, feather (soft transition), refine (edge-aware snap to the
    picture's real edges) and strength (opacity). Feathering is always re-anchored to the image with a guided
    filter, so a soft edge never bleeds across a hard edge (hair, mountain ridges): no halos;
  * everything is cached per (mask, controls, size) so repeated edits of the same image are cheap.

A semantic mask is not automatically right for every operation: a person mask is not a skin mask, so they are kept
separate and an adjustment chooses the one that fits (skin for skin tone, person for subject separation).
"""
from typing import Dict, Iterable, Optional

import cv2
import numpy as np

from landscape import _guided

DERIVED = ("foreground", "background")
SCENE_REGIONS = ("sky", "mountains", "trees", "grass", "water", "buildings", "ground", "person")
OTHER_MIN_AREA = 0.005     # below this the picture is fully covered and no "Miscellaneous" region is offered


# ---------------------------------------------------------------------------------------------
# soft set operations (all arrays float32 in 0..1)
# ---------------------------------------------------------------------------------------------
def invert(m: np.ndarray) -> np.ndarray:
    return 1.0 - m


def union(*masks: np.ndarray) -> np.ndarray:
    out = np.ones_like(masks[0])
    for m in masks:
        out = out * (1.0 - m)
    return 1.0 - out


def intersect(*masks: np.ndarray) -> np.ndarray:
    out = masks[0].copy()
    for m in masks[1:]:
        out = out * m
    return out


def subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(a - b, 0.0, 1.0)


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)


class MaskBank:
    """Masks for one analysed photo. Masks are stored at working size as float16 and resized on demand."""

    def __init__(self, masks: Dict[str, np.ndarray], depth: Optional[np.ndarray] = None):
        self.base = masks
        self.depth = depth
        self._cache: Dict[tuple, np.ndarray] = {}
        self._other = self._compute_other()

    # ------------------------------------------------------------------ availability
    def _compute_other(self) -> Optional[np.ndarray]:
        """Everything no scene region claimed ("Miscellaneous"), so every pixel of the picture belongs to something:
        raw leftover = 1 - the strongest claim on each pixel; speckle is removed and the edge softened."""
        got = [self.base[k].astype(np.float32) for k in SCENE_REGIONS if k in self.base]
        if not got:
            return None
        raw = np.clip(1.0 - np.max(got, axis=0), 0.0, 1.0)
        binary = (raw > 0.5).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros_like(binary)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= 0.0008 * binary.size:      # tiny leftovers are noise in the masks, not objects
                keep[lab == i] = 1
        soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), max(1.0, max(raw.shape) / 400))
        out = np.clip(np.maximum(soft, keep) * raw * 1.6, 0, 1)
        return out.astype(np.float16) if float((out > 0.5).mean()) >= OTHER_MIN_AREA else None

    def coverage(self) -> Dict[str, float]:
        """Percentage of the picture each region owns, plus the share left as Miscellaneous (should be small)."""
        cov = {k: round(float((self.base[k] > 0.5).mean() * 100), 1) for k in SCENE_REGIONS if k in self.base}
        cov["other"] = round(float((self._other > 0.5).mean() * 100), 1) if self._other is not None else 0.0
        return cov

    def names(self) -> list:
        out = [k for k in self.base if k not in ("hair", "face")]
        if "person" in self.base or self.depth is not None:
            out += list(DERIVED)
        if self._other is not None:
            out.append("other")
        return out

    def has(self, name: str) -> bool:
        return name in self.names() or name == "hair" and "hair" in self.base

    # ------------------------------------------------------------------ plain and derived masks
    def get(self, name: str, size: tuple) -> Optional[np.ndarray]:
        """The mask at `size` = (w, h), float32 0..1, or None when it does not exist for this photo."""
        key = ("raw", name, size)
        if key in self._cache:
            return self._cache[key]
        m = self._build(name, size)
        if m is not None:
            self._cache[key] = m
        return m

    def _resize(self, a: np.ndarray, size: tuple) -> np.ndarray:
        a = a.astype(np.float32)
        w, h = size
        if a.shape[1] == w and a.shape[0] == h:
            return np.clip(a, 0, 1)
        return np.clip(cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA if w < a.shape[1] else cv2.INTER_CUBIC), 0, 1)

    def _build(self, name: str, size: tuple) -> Optional[np.ndarray]:
        if name in self.base:
            return self._resize(self.base[name], size)
        if name == "other":
            return None if self._other is None else self._resize(self._other, size)
        if name in DERIVED:
            fg = None
            if "person" in self.base:
                fg = self._resize(self.base["person"], size)
            elif self.depth is not None:      # no person: the nearest part of the scene is the foreground
                d = self._resize(self.depth, size)
                sky = self._resize(self.base["sky"], size) if "sky" in self.base else np.zeros_like(d)
                land = sky < 0.3
                lo, hi = np.percentile(d[land] if land.sum() > 100 else d, [55, 92])
                fg = np.clip((d - lo) / max(hi - lo, 1e-3), 0, 1) ** 1.2 * (1.0 - sky)
                fg = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), max(size) / 220)
            if fg is None:
                return None
            return fg if name == "foreground" else 1.0 - fg
        return None

    # ------------------------------------------------------------------ the user's controls
    def effective(self, name: str, size: tuple, guide: np.ndarray, strength: float = 100.0, feather: float = 0.0,
                  refine: float = 0.0, invert_mask: bool = False) -> Optional[np.ndarray]:
        """The mask with the user's controls applied (all 0..100). `guide` is the picture at `size`, used to keep
        soft edges attached to real edges."""
        key = ("eff", name, size, round(strength, 1), round(feather, 1), round(refine, 1), bool(invert_mask))
        if key in self._cache:
            return self._cache[key]
        m = self.get(name, size)
        if m is None:
            return None
        m = self.shape(m, guide, strength, feather, refine, invert_mask)
        self._cache[key] = m
        return m

    def shape(self, m: np.ndarray, guide: np.ndarray, strength: float = 100.0, feather: float = 0.0,
              refine: float = 0.0, invert_mask: bool = False) -> np.ndarray:
        """invert -> feather -> edge-aware refine -> strength. Works on any mask, including combined ones."""
        if invert_mask:
            m = 1.0 - m
        edge = max(m.shape[:2])
        if feather > 0:
            sigma = edge * (0.0012 + 0.018 * feather / 100)
            m = cv2.GaussianBlur(m, (0, 0), sigma)
        anchor = max(refine / 100, 0.55 if feather > 0 else 0.0)   # feathered masks are always re-anchored
        if anchor > 0:
            g = _gray(guide)
            s = min(1.0, 900 / edge)
            gs = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
            ms = cv2.resize(m, (gs.shape[1], gs.shape[0]), interpolation=cv2.INTER_AREA) if s < 1 else m
            r = max(2, int(max(gs.shape) / (160 - 100 * min(1.0, refine / 100))))
            snapped = _guided(gs, ms.astype(np.float32), r=r, eps=1e-3 * (1.0 - 0.8 * refine / 100))
            snapped = cv2.resize(snapped, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_CUBIC) if s < 1 else snapped
            m = m * (1 - anchor) + np.clip(snapped, 0, 1) * anchor
        return np.clip(m * (strength / 100.0), 0.0, 1.0).astype(np.float32)

    def clear(self):
        self._cache.clear()
