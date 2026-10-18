"""Scene analysis: face landmarks, a face-focus check, sky detection and scene type.

Everything here runs locally (MediaPipe face landmarks + OpenCV), so it costs nothing per photo.
"""
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from landscape import _guided

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"
DETECT_EDGE = 1600  # landmarks are found on a downscaled copy, then scaled back up

_C = vision.FaceLandmarksConnections


def _idx(conns) -> List[int]:
    return sorted({c.start for c in conns} | {c.end for c in conns})


OVAL = _idx(_C.FACE_LANDMARKS_FACE_OVAL)
LEFT_EYE = _idx(_C.FACE_LANDMARKS_LEFT_EYE)
RIGHT_EYE = _idx(_C.FACE_LANDMARKS_RIGHT_EYE)
LEFT_BROW = _idx(_C.FACE_LANDMARKS_LEFT_EYEBROW)
RIGHT_BROW = _idx(_C.FACE_LANDMARKS_RIGHT_EYEBROW)
LEFT_IRIS = _idx(_C.FACE_LANDMARKS_LEFT_IRIS)
RIGHT_IRIS = _idx(_C.FACE_LANDMARKS_RIGHT_IRIS)
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

SEG_PATH = Path(__file__).parent / "models" / "selfie_multiclass.tflite"
SEG_EDGE = 1024  # segmentation runs at reduced size; edges are then refined against the real image

_detector = None
_segmenter = None
_lock = threading.Lock()


def _get_detector():
    global _detector
    if _detector is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing {MODEL_PATH}. Download face_landmarker.task into models/.")
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)), num_faces=6)
        _detector = vision.FaceLandmarker.create_from_options(opts)
    return _detector


def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        if not SEG_PATH.exists():
            raise FileNotFoundError(f"Missing {SEG_PATH}. Download selfie_multiclass_256x256.tflite into models/.")
        _segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(SEG_PATH)),
            output_category_mask=False, output_confidence_masks=True))
    return _segmenter


@dataclass
class Person:
    mask: np.ndarray        # float32 HxW, whole person (hair, skin, clothes, accessories)
    skin: np.ndarray        # face + body skin
    hair: np.ndarray
    frac: float


def segment_person(rgb: np.ndarray) -> Optional[Person]:
    """Whole-person mask. Classes: 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 other."""
    h, w = rgb.shape[:2]
    scale = min(1.0, SEG_EDGE / max(h, w))
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    small8 = np.ascontiguousarray((np.clip(small, 0, 1) * 255).astype(np.uint8))
    with _lock:
        res = _get_segmenter().segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=small8))
    conf = [np.asarray(m.numpy_view(), dtype=np.float32).reshape(small8.shape[:2]) for m in res.confidence_masks]
    person = np.clip(1 - conf[0], 0, 1)
    frac = float(person.mean())
    if frac < 0.01:
        return None
    guide = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = max(3, max(small8.shape[:2]) // 100)

    def refine(m, full=True):
        m = np.clip(_guided(guide, m, r=r, eps=1e-3), 0, 1)
        # Low-confidence leakage (a constant few % everywhere) would blend the un-graded person
        # version into the background and show up as streaks: snap it to exactly 0 / 1.
        m = np.clip((m - 0.15) / 0.70, 0, 1)
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1) if scale < 1 or full else m

    return Person(mask=refine(person), skin=refine(np.clip(conf[2] + conf[3], 0, 1)),
                  hair=refine(conf[1]), frac=frac)


@dataclass
class Face:
    pts: np.ndarray            # (478, 2) pixel coordinates in the full-resolution image
    area_frac: float           # face oval area / image area
    width: float               # face width in pixels
    focus_score: float = 0.0
    focus: str = "unknown"     # sharp | soft | blurry

    def poly(self, idx: List[int]) -> np.ndarray:
        return self.pts[idx].astype(np.int32)


@dataclass
class Scene:
    kind: str                  # portrait | landscape | general
    faces: List[Face] = field(default_factory=list)
    sky_frac: float = 0.0
    sky_mask: Optional[np.ndarray] = None   # float32 HxW at full resolution (feathered), or None
    haze: float = 0.0          # 0 (clear) .. 1 (very hazy)
    person: Optional[Person] = None

    def summary(self) -> dict:
        return {
            "scene": self.kind,
            "faces": [{"focus": f.focus, "focus_score": round(f.focus_score, 1),
                       "size_pct": round(f.area_frac * 100, 1)} for f in self.faces],
            "sky_pct": round(self.sky_frac * 100, 1),
            "person_pct": round(self.person.frac * 100, 1) if self.person else 0.0,
            "haze": round(self.haze, 2),
        }


def _focus(gray_full: np.ndarray, face: Face) -> None:
    """Sharpness of the eye region, normalised so face size/resolution don't matter."""
    eyes = np.vstack([face.pts[LEFT_EYE], face.pts[RIGHT_EYE], face.pts[LEFT_BROW], face.pts[RIGHT_BROW]])
    iod = np.linalg.norm(face.pts[LEFT_EYE].mean(0) - face.pts[RIGHT_EYE].mean(0))
    if iod < 35:  # too few pixels on the eyes to judge focus reliably
        face.focus = "n/a"
        return
    pad = iod * 0.25
    x0, y0 = np.maximum(eyes.min(0) - pad, 0).astype(int)
    x1, y1 = (eyes.max(0) + pad).astype(int)
    crop = gray_full[y0:y1, x0:x1]
    if crop.size == 0:
        return
    scale = 120.0 / iod  # inter-eye distance -> 120 px
    crop = cv2.resize(crop, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    face.focus_score = float(cv2.Laplacian(crop, cv2.CV_32F).var())
    face.focus = "sharp" if face.focus_score >= 30 else "soft" if face.focus_score >= 8 else "blurry"


def _detect_crop(rgb: np.ndarray, box, edge: int) -> List[np.ndarray]:
    """Run the landmarker on a crop (x0, y0, x1, y1) of the image; return landmarks in full-image pixels."""
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    scale = min(1.0, edge / max(ch, cw))
    small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else crop
    small8 = np.ascontiguousarray((np.clip(small, 0, 1) * 255).astype(np.uint8))
    with _lock:
        result = _get_detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=small8))
    return [np.array([[x0 + p.x * cw, y0 + p.y * ch] for p in lm], dtype=np.float32) for lm in result.face_landmarks]


def _tiles(h: int, w: int, frac: float):
    th, tw = int(h * frac), int(w * frac)
    step_y, step_x = max(1, th // 2), max(1, tw // 2)
    ys = list(range(0, max(1, h - th + 1), step_y)) or [0]
    xs = list(range(0, max(1, w - tw + 1), step_x)) or [0]
    if ys[-1] + th < h:
        ys.append(h - th)
    if xs[-1] + tw < w:
        xs.append(w - tw)
    return [(x, y, min(w, x + tw), min(h, y + th)) for y in ys for x in xs]


def detect_faces(rgb: np.ndarray) -> List[Face]:
    """The landmarker's detector has a fixed input size, so it only sees faces within a limited size
    range. Search the whole frame at several resolutions, then zoomed overlapping tiles for small
    faces (group and travel shots), and merge duplicates."""
    h, w = rgb.shape[:2]
    full = (0, 0, w, h)
    passes = [(full, 640), (full, 1000), (full, 1600)]
    passes += [(b, 960) for b in _tiles(h, w, 0.5)] + [(b, 960) for b in _tiles(h, w, 0.3)]

    found: List[np.ndarray] = []
    for box, edge in passes:
        for pts in _detect_crop(rgb, box, edge):
            if (pts < 0).any() or (pts[:, 0] > w).any() or (pts[:, 1] > h).any():
                continue
            c, fw = pts[OVAL].mean(0), np.ptp(pts[OVAL][:, 0])
            if not any(np.linalg.norm(c - q[OVAL].mean(0)) < 0.6 * max(fw, np.ptp(q[OVAL][:, 0])) for q in found):
                found.append(pts)

    gray_full = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) * 255).astype(np.float32)
    faces = []
    for pts in found:
        area = cv2.contourArea(cv2.convexHull(pts[OVAL].astype(np.float32)))
        face = Face(pts=pts, area_frac=area / (w * h), width=float(np.ptp(pts[OVAL][:, 0])))
        _focus(gray_full, face)
        faces.append(face)
    faces.sort(key=lambda f: -f.area_frac)
    return faces


def detect_sky(rgb: np.ndarray, exclude: Optional[np.ndarray] = None) -> tuple:
    """Heuristic sky mask: smooth, bright or blue regions connected to the top of the frame.

    Returns (feathered float32 mask at full resolution, fraction of image that is sky).
    """
    h, w = rgb.shape[:2]
    scale = min(1.0, 640 / max(h, w))
    t = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    th, tw = t.shape[:2]
    hsv = cv2.cvtColor(t, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    gray = cv2.GaussianBlur(cv2.cvtColor(t, cv2.COLOR_RGB2GRAY), (0, 0), 2)
    grad = cv2.GaussianBlur(np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)), (0, 0), 3)
    smooth = grad < 0.06

    blue = (hue > 185) & (hue < 255) & (sat > 0.12) & (val > 0.25)
    overcast = (sat < 0.22) & (val > 0.55)
    warm = (sat > 0.2) & (val > 0.45) & ((hue < 50) | (hue > 330)) & smooth  # sunrise/sunset glow
    # Sunset skies pass through teal / yellow-green between the blue and the glow: bright and smooth.
    horizon = (hue >= 45) & (hue <= 185) & (val > 0.65) & (sat < 0.6) & smooth
    cand = ((blue | overcast) & smooth) | warm | horizon
    if exclude is not None:
        ex = cv2.resize(exclude, (tw, th), interpolation=cv2.INTER_AREA) > 0.25
        cand = cand & ~ex
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, labels = cv2.connectedComponents(cand)
    sky = np.zeros_like(cand)
    top_band = labels[: max(2, th // 25)]
    for lab in np.unique(top_band):
        if lab == 0:
            continue
        if (labels == lab).sum() > 0.01 * th * tw:
            sky[labels == lab] = 1
    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    frac = float(sky.mean())
    if frac < 0.06:
        return None, frac

    # The smoothness test leaves a band of "not sky" around every object (gradients are high there).
    # Grow the mask back into sky-coloured pixels, then snap it to the real edges with a guided filter.
    color_ok = (blue | overcast | ((sat > 0.2) & (val > 0.45) & ((hue < 50) | (hue > 330)))
                | ((hue >= 45) & (hue <= 185) & (val > 0.65) & (sat < 0.6))).astype(np.uint8)
    if exclude is not None:
        color_ok = color_ok & ~ex
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for _ in range(4):
        sky = (cv2.dilate(sky, grow) & color_ok) | sky
    guide = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY).astype(np.float32)
    soft = np.clip(_guided(guide, cv2.GaussianBlur(sky.astype(np.float32), (0, 0), 1.5), r=max(3, tw // 80), eps=2e-3), 0, 1)
    out = cv2.resize(soft, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    if exclude is not None:
        out = out * (1 - np.clip(exclude, 0, 1))
    return out, frac


def estimate_haze(rgb: np.ndarray, sky_mask: Optional[np.ndarray] = None) -> float:
    """Dark-channel statistic on non-sky pixels: clear scenes have near-black pixels in most patches."""
    scale = min(1.0, 400 / max(rgb.shape[:2]))
    t = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    dark = cv2.erode(t.min(axis=2), np.ones((9, 9), np.uint8))
    keep = np.ones(dark.shape, bool)
    if sky_mask is not None:
        keep = cv2.resize(sky_mask, (dark.shape[1], dark.shape[0])) < 0.3
    if keep.mean() < 0.1:
        return 0.0
    return float(np.clip((np.percentile(dark[keep], 35) - 0.07) / 0.30, 0, 1))


def is_screenshot(rgb: np.ndarray) -> bool:
    """Screenshots / documents: one flat colour fills most of the frame (photos never do)."""
    scale = min(1.0, 500 / max(rgb.shape[:2]))
    t = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    q = (np.clip(t, 0, 1) * 63).astype(np.int32)
    packed = (q[..., 0] << 12) | (q[..., 1] << 6) | q[..., 2]
    return float(np.bincount(packed.ravel()).max() / packed.size) > 0.30


def analyze_scene(rgb: np.ndarray) -> Scene:
    if is_screenshot(rgb):
        return Scene(kind="screenshot")
    faces = detect_faces(rgb)
    person = segment_person(rgb) if faces else None
    sky_mask, sky_frac = detect_sky(rgb, person.mask if person else None)
    haze = estimate_haze(rgb, sky_mask)
    biggest = faces[0].area_frac if faces else 0.0
    if biggest >= 0.015:
        kind = "portrait"            # face is the subject, close to the lens
    elif biggest >= 0.001:
        kind = "people"              # travel / group shot: scenery recipe + gentle person care
    elif sky_frac >= 0.10 or haze > 0.35:
        kind = "landscape"
    else:
        kind = "general"
    faces = [f for f in faces if f.area_frac >= 0.001]
    return Scene(kind=kind, faces=faces, sky_frac=sky_frac, sky_mask=sky_mask, haze=haze, person=person)
