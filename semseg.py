"""Scene segmentation (SegFormer-B2, ADE20K, 150 classes) run locally with ONNX Runtime.

Tells the engine what each pixel *is* (sky, tree, water, building, ground, person, indoor furniture),
which fixes the mistakes colour-based masks make: sea mistaken for sky, a blue wall mistaken for sky,
a green shirt mistaken for foliage. Returns soft probability maps per region group.
"""
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

MODEL = Path(__file__).parent / "models" / "segformer_b2_ade.onnx"
CONFIG = Path(__file__).parent / "models" / "segformer_ade_config.json"
LONG_EDGE = 768
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

GROUPS = {
    "sky": ["sky"],
    "trees": ["tree", "palm", "plant", "flower"],
    "grass": ["grass", "field"],
    "water": ["water", "sea", "river", "lake", "waterfall", "swimming pool", "fountain"],
    "mountains": ["mountain", "hill", "rock"],
    "buildings": ["building", "house", "skyscraper", "tower", "hovel", "bridge", "pier"],
    "ground": ["earth", "land", "sand", "dirt track", "road", "sidewalk", "path"],
    "person": ["person"],
}
INDOOR = ["ceiling", "bed", "cabinet", "sofa", "curtain", "shelf", "rug", "armchair", "wardrobe", "lamp", "desk",
          "counter", "sink", "bookcase", "kitchen island", "chandelier", "refrigerator", "stove", "oven", "bathtub",
          "toilet", "coffee table", "table", "chair", "swivel chair", "stool", "ottoman", "pillow", "cushion"]

_session = None
_ids: Dict[str, list] = {}
_indoor_ids: list = []
_lock = threading.Lock()


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return MODEL.exists() and CONFIG.exists()


def _load():
    global _session, _ids, _indoor_ids
    if _session is not None:
        return
    import onnxruntime as ort
    labels = {int(k): v.strip() for k, v in json.load(open(CONFIG))["id2label"].items()}
    by_name = {v: k for k, v in labels.items()}
    _ids = {g: [by_name[n] for n in names if n in by_name] for g, names in GROUPS.items()}
    _indoor_ids = [by_name[n] for n in INDOOR if n in by_name]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 0
    _session = ort.InferenceSession(str(MODEL), sess_options=opts, providers=["CPUExecutionProvider"])


@dataclass
class SemResult:
    groups: Dict[str, np.ndarray] = field(default_factory=dict)   # float32 HxW probabilities (work size)
    shares: Dict[str, float] = field(default_factory=dict)        # fraction of the frame per group
    indoor: bool = False

    def summary(self) -> dict:
        return {"shares": {k: round(v * 100, 1) for k, v in self.shares.items()}, "indoor": self.indoor}


def _infer(img: np.ndarray) -> np.ndarray:
    x = ((img - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    with _lock:
        return _session.run(None, {"pixel_values": x})[0][0]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


def _scale_probs(rgb: np.ndarray, long_edge: int, flip: bool) -> np.ndarray:
    """Class probabilities (150 x h/4 x w/4) at one input scale, optionally mirror-averaged."""
    h, w = rgb.shape[:2]
    s = long_edge / max(h, w)
    nw, nh = max(32, int(round(w * s / 32)) * 32), max(32, int(round(h * s / 32)) * 32)
    img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    probs = _softmax(_infer(img))
    if flip:
        probs = 0.5 * (probs + _softmax(_infer(img[:, ::-1].copy()))[:, :, ::-1])
    return probs


def run(rgb: np.ndarray, scales=((640, True), (1024, False))) -> Optional[SemResult]:
    """rgb: float32 [0,1]. Multi-scale (coarse context + fine edges) probability maps per group, at the
    input size. Returns None if the model or runtime is unavailable."""
    if not available():
        return None
    _load()
    h, w = rgb.shape[:2]
    res = SemResult()
    acc = None
    label = None
    for edge, flip in scales:
        p = _scale_probs(rgb, edge, flip)
        # bring every scale to a common grid before averaging
        if acc is None:
            acc, gh, gw = p, p.shape[1], p.shape[2]
        else:
            p = np.stack([cv2.resize(c, (gw, gh), interpolation=cv2.INTER_LINEAR) for c in p])
            acc = acc + p
    acc = acc / len(scales)
    label = acc.argmax(axis=0)
    for g, ids in _ids.items():
        pg = acc[ids].sum(axis=0).astype(np.float32)
        res.groups[g] = np.clip(cv2.resize(pg, (w, h), interpolation=cv2.INTER_LINEAR), 0, 1)
        res.shares[g] = float(np.isin(label, ids).mean())
    res.shares["indoor_objects"] = float(np.isin(label, _indoor_ids).mean())
    res.indoor = res.shares["indoor_objects"] > 0.14 and res.shares["sky"] < 0.04
    return res
