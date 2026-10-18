"""3D LUTs: import any .cube file, apply it with trilinear interpolation, and a library of built-in looks.

A .cube file is a 3-D colour lookup table: for every input (R, G, B) it stores the output colour. Colourists share looks
as .cube files, so supporting the format means any LUT from a film emulation pack or a colourist works here.
Built-in looks are generated procedurally (so they ship with the app) into the same 33x33x33 table form, which means a
built-in and an imported LUT go through exactly the same code.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

LUT_DIR = Path(__file__).parent / "luts"
LUT_DIR.mkdir(exist_ok=True)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
MAX_CUBE_BYTES = 12 * 1024 * 1024


@dataclass
class Lut:
    id: str
    name: str
    table: np.ndarray               # (N, N, N, 3), indexed [b, g, r], values 0..1
    builtin: bool = False
    domain: tuple = (0.0, 1.0)

    @property
    def size(self) -> int:
        return int(self.table.shape[0])


# ------------------------------------------------------------------------------------------------ .cube files
def parse_cube(text: str, name_hint: str = "LUT") -> Lut:
    """Parse an Adobe/Resolve .cube file (3D). 1D LUTs are expanded to 3D. Raises ValueError on a malformed file."""
    size3 = size1 = None
    dmin, dmax = 0.0, 1.0
    title = name_hint
    rows: List[List[float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        up = line.upper()
        if up.startswith("TITLE"):
            m = re.search(r'"(.*)"', line)
            title = (m.group(1) if m else line[5:]).strip() or name_hint
        elif up.startswith("LUT_3D_SIZE"):
            size3 = int(line.split()[1])
        elif up.startswith("LUT_1D_SIZE"):
            size1 = int(line.split()[1])
        elif up.startswith("DOMAIN_MIN"):
            dmin = float(line.split()[1])
        elif up.startswith("DOMAIN_MAX"):
            dmax = float(line.split()[1])
        elif up.startswith(("LUT_", "DOMAIN_")):
            continue
        else:
            parts = line.split()
            if len(parts) == 3:
                try:
                    rows.append([float(v) for v in parts])
                except ValueError:
                    raise ValueError(f"Unreadable line in the .cube file: {line[:40]!r}")
    data = np.array(rows, np.float32)
    span = max(dmax - dmin, 1e-6)
    if size3:
        if len(data) != size3 ** 3:
            raise ValueError(f"The .cube file says {size3}x{size3}x{size3} but has {len(data)} colour rows.")
        if size3 < 2 or size3 > 129:
            raise ValueError("Unsupported LUT size.")
        table = ((data - dmin) / span).reshape(size3, size3, size3, 3)          # file order: red changes fastest -> [b][g][r]
    elif size1:
        if len(data) != size1 or size1 < 2:
            raise ValueError("Malformed 1D LUT.")
        curve = np.clip((data - dmin) / span, 0, 1)
        g = np.linspace(0, 1, 33, dtype=np.float32)
        xs = np.linspace(0, 1, size1, dtype=np.float32)
        ch = [np.interp(g, xs, curve[:, c]) for c in range(3)]
        table = np.zeros((33, 33, 33, 3), np.float32)
        table[..., 0] = ch[0][None, None, :]
        table[..., 1] = ch[1][None, :, None]
        table[..., 2] = ch[2][:, None, None]
    else:
        raise ValueError("This does not look like a .cube LUT (no LUT_3D_SIZE).")
    return Lut(id="", name=title, table=np.clip(table, 0.0, 1.0), builtin=False)


def to_cube(lut: Lut) -> str:
    n = lut.size
    out = [f'TITLE "{lut.name}"', f"LUT_3D_SIZE {n}"]
    flat = lut.table.reshape(-1, 3)
    out += [f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in flat]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ applying
def apply(arr: np.ndarray, lut: Lut, strength: float = 100.0) -> np.ndarray:
    """Trilinear lookup. strength 0..100 blends from the original (0) to the full look (100). Works in row chunks so a
    24-megapixel export does not need gigabytes of temporary memory; uses one flat gather per cube corner (fast)."""
    if strength <= 0:
        return arr
    n = lut.size
    flat = np.ascontiguousarray(lut.table.reshape(-1, 3))
    h, w = arr.shape[:2]
    out = np.empty((h, w, 3), np.float32)
    rows = max(1, int(1_500_000 // max(w, 1)))
    k = np.float32(strength / 100.0)
    s1, s2 = n, n * n                                  # flat strides for [b][g][r]
    for y0 in range(0, h, rows):
        a = arr[y0:y0 + rows]
        x = np.clip(a, 0, 1) * np.float32(n - 1)
        i0 = np.minimum(x.astype(np.int32), n - 2)
        f = x - i0
        base = i0[..., 0] + i0[..., 1] * s1 + i0[..., 2] * s2
        fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
        c = [flat.take(base + (dr + dg * s1 + db * s2), axis=0) for db in (0, 1) for dg in (0, 1) for dr in (0, 1)]
        c00 = c[0] + (c[1] - c[0]) * fr
        c10 = c[2] + (c[3] - c[2]) * fr
        c01 = c[4] + (c[5] - c[4]) * fr
        c11 = c[6] + (c[7] - c[6]) * fr
        c0 = c00 + (c10 - c00) * fg
        c1 = c01 + (c11 - c01) * fg
        res = c0 + (c1 - c0) * fb
        out[y0:y0 + rows] = (a + (res - a) * k) if k < 1 else res
    return np.clip(out, 0, 1, out=out)


# ------------------------------------------------------------------------------------------------ built-in looks
def _sat(c: np.ndarray, f: float) -> np.ndarray:
    y = (c @ LUMA)[..., None]
    return y + (c - y) * f


def _contrast(c: np.ndarray, k: float) -> np.ndarray:
    s = c * c * (3 - 2 * c)
    return c + (s - c) * k


def _split(c: np.ndarray, shadow, high, amount: float) -> np.ndarray:
    y = (c @ LUMA)[..., None]
    sh = (1 - y) ** 1.6
    hi = y ** 1.6
    return c + amount * (sh * (np.array(shadow, np.float32) - 0.5) + hi * (np.array(high, np.float32) - 0.5))


def _lift(c: np.ndarray, black: float, white: float = 1.0) -> np.ndarray:
    return black + c * (white - black)


def _hue_sat(c: np.ndarray, center: float, width: float, f: float) -> np.ndarray:
    mx, mn = c.max(axis=-1), c.min(axis=-1)
    d = np.maximum(mx - mn, 1e-6)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
    w = np.clip(1 - np.abs((h - center + 180) % 360 - 180) / width, 0, 1)[..., None]
    y = (c @ LUMA)[..., None]
    return y + (c - y) * (1 + (f - 1) * w)


def _bw(c: np.ndarray, contrast: float, tint=(1.0, 1.0, 1.0)) -> np.ndarray:
    y = (c @ np.array([0.30, 0.59, 0.11], np.float32))[..., None]
    y = _contrast(np.clip(y, 0, 1), contrast)
    return y * np.array(tint, np.float32)


BUILTIN: Dict[str, tuple] = {
    "teal_orange": ("Teal & Orange", lambda c: _sat(_contrast(_split(c, (0.15, 0.55, 0.62), (0.95, 0.62, 0.30), 0.55), 0.35), 1.05)),
    "golden_film": ("Golden Film", lambda c: _lift(_sat(_split(c, (0.42, 0.50, 0.55), (0.98, 0.78, 0.48), 0.5), 1.05), 0.035, 0.98)),
    "blue_hour": ("Cool Blue Hour", lambda c: _sat(_contrast(_split(c, (0.30, 0.42, 0.78), (0.78, 0.82, 0.92), 0.5), 0.2), 0.92)),
    "faded_matte": ("Faded Matte", lambda c: _sat(_lift(_contrast(c, -0.1), 0.085, 0.96), 0.86)),
    "rich_landscape": ("Rich Landscape", lambda c: _hue_sat(_hue_sat(_sat(_contrast(c, 0.4), 1.1), 110, 60, 1.18), 220, 50, 1.15)),
    "moody_forest": ("Moody Forest", lambda c: _sat(_contrast(_split(c, (0.20, 0.42, 0.40), (0.78, 0.74, 0.55), 0.5), 0.3) * 0.94, 0.88)),
    "soft_portrait": ("Warm Portrait", lambda c: _lift(_sat(_split(c, (0.46, 0.47, 0.52), (1.0, 0.86, 0.72), 0.4), 0.96), 0.02, 0.99)),
    "vivid_pop": ("Vivid Pop", lambda c: _sat(_contrast(c, 0.3), 1.28)),
    "bw_classic": ("B&W Classic", lambda c: _bw(c, 0.25)),
    "bw_contrast": ("B&W High Contrast", lambda c: _bw(c, 0.7)),
    "bw_warm": ("B&W Warm Tone", lambda c: _bw(c, 0.3, (1.0, 0.94, 0.84))),
}

_cache: Dict[str, Lut] = {}


def _grid(n: int = 33) -> np.ndarray:
    g = np.linspace(0, 1, n, dtype=np.float32)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([r, gg, b], axis=-1)           # [b, g, r] order, last axis = (R, G, B)


def _sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._ -]+", "", Path(name).stem).strip().replace(" ", "_")
    return s[:60] or "lut"


def list_luts() -> List[dict]:
    out = [{"id": f"builtin:{k}", "name": v[0], "builtin": True, "size": 33} for k, v in BUILTIN.items()]
    for f in sorted(LUT_DIR.glob("*.cube")):
        out.append({"id": f"user:{f.stem}", "name": f.stem.replace("_", " "), "builtin": False, "size": 0})
    return out


def get(lut_id: str) -> Optional[Lut]:
    if lut_id in _cache:
        return _cache[lut_id]
    kind, _, key = lut_id.partition(":")
    lut = None
    if kind == "builtin" and key in BUILTIN:
        name, fn = BUILTIN[key]
        lut = Lut(id=lut_id, name=name, table=np.clip(fn(_grid()), 0, 1).astype(np.float32), builtin=True)
    elif kind == "user":
        path = LUT_DIR / f"{_sanitize(key)}.cube"
        if path.exists():
            lut = parse_cube(path.read_text(errors="ignore"), key)
            lut.id, lut.name = lut_id, key.replace("_", " ")
    if lut is not None:
        _cache[lut_id] = lut
    return lut


def save_user(filename: str, data: bytes) -> str:
    """Validate and store an uploaded .cube. Returns the new LUT id. Raises ValueError for anything that is not a usable LUT."""
    if len(data) > MAX_CUBE_BYTES:
        raise ValueError("That LUT file is too large (max 12 MB).")
    lut = parse_cube(data.decode("utf-8", errors="ignore"), Path(filename).stem)
    key = _sanitize(filename)
    (LUT_DIR / f"{key}.cube").write_bytes(data)
    _cache.pop(f"user:{key}", None)
    return f"user:{key}"


def delete_user(lut_id: str) -> bool:
    kind, _, key = lut_id.partition(":")
    if kind != "user":
        return False
    path = LUT_DIR / f"{_sanitize(key)}.cube"
    _cache.pop(lut_id, None)
    if path.exists():
        path.unlink()
        return True
    return False
