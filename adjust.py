"""Local, deterministic adjustment engine (the "Lightroom" part of the pipeline).

Runs at the photo's original resolution, keeps EXIF and the ICC profile, and never
regenerates pixels. `analyze()` proposes slider values from the image's own statistics;
`apply()` renders any set of slider values. Later, a vision model can propose the
`Adjustments` instead of (or on top of) `analyze()` without touching the engine.
"""
import io
import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageFilter, ImageOps
from pillow_heif import register_heif_opener
from pydantic import BaseModel, Field

register_heif_opener()  # iPhone HEIC/HEIF support
Image.MAX_IMAGE_PIXELS = None  # large DSLR / stitched files are expected

GAMMA = 2.2  # sRGB approximation, fast and close enough for grading
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


class Adjustments(BaseModel):
    """Slider values. Ranges mirror Lightroom conventions."""
    exposure: float = Field(0, ge=-3, le=3)        # EV stops
    temperature: float = Field(0, ge=-100, le=100)  # + warmer
    tint: float = Field(0, ge=-100, le=100)         # + magenta
    contrast: float = Field(0, ge=-100, le=100)
    highlights: float = Field(0, ge=-100, le=100)   # - recovers highlights
    shadows: float = Field(0, ge=-100, le=100)      # + lifts shadows
    vibrance: float = Field(0, ge=-100, le=100)     # boosts muted colours more
    saturation: float = Field(0, ge=-100, le=100)
    clarity: float = Field(0, ge=-100, le=100)      # midtone local contrast
    sharpen: float = Field(0, ge=0, le=100)

    def __add__(self, other: "Adjustments") -> "Adjustments":
        merged = {}
        for name, field in Adjustments.model_fields.items():
            lo = next(m.ge for m in field.metadata if hasattr(m, "ge"))
            hi = next(m.le for m in field.metadata if hasattr(m, "le"))
            merged[name] = min(hi, max(lo, getattr(self, name) + getattr(other, name)))
        return Adjustments(**merged)


# Looks are offsets on top of the auto-analysis, so every photo still gets corrected first.
PRESETS = {
    "natural": Adjustments(vibrance=8, clarity=8, sharpen=25),
    "vivid": Adjustments(contrast=12, vibrance=25, saturation=8, clarity=15, sharpen=35),
    "cinematic": Adjustments(contrast=22, highlights=-15, shadows=-8, saturation=-12, clarity=10, sharpen=25),
    "soft_portrait": Adjustments(contrast=-5, vibrance=10, clarity=-12, shadows=12, sharpen=15),
}


RAW_EXT = {".nef", ".nrw", ".dng", ".cr2", ".cr3", ".crw", ".arw", ".sr2", ".raf", ".orf", ".rw2", ".pef", ".srw", ".3fr", ".iiq"}


def is_raw(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXT


def _load_raw(path: str, max_edge: Optional[int]) -> Image.Image:
    """Camera raw (NEF, DNG, ARW, CR2, ...) via LibRaw. With max_edge, uses the fast half-size decode."""
    import io
    import rawpy

    with rawpy.imread(path) as r:
        half = bool(max_edge) and max(r.sizes.width, r.sizes.height) >= 2 * max_edge
        rgb = r.postprocess(half_size=half, use_camera_wb=True, output_bps=8, output_color=rawpy.ColorSpace.sRGB,
                            highlight_mode=rawpy.HighlightMode.Blend)  # Blend recovers clipped highlights
        exif = None
        try:  # the embedded preview JPEG carries the camera's EXIF (lens, ISO, date, ...)
            th = r.extract_thumb()
            if th.format == rawpy.ThumbFormat.JPEG:
                ex = Image.open(io.BytesIO(th.data)).getexif()
                ex[0x0112] = 1  # LibRaw already rotated the pixels
                exif = ex.tobytes()
        except Exception:
            pass
    img = Image.fromarray(rgb)
    img.info["exif"], img.info["icc_profile"] = exif, None  # sRGB output
    return img


def load_float(path: str, max_edge: Optional[int] = None) -> np.ndarray:
    """The photo as float32 RGB in 0..1, ready for editing. Camera raw files are developed at 16 bits and kept
    that way, so shadow lifts, highlight recovery and dodge & burn have the sensor's full tonal range to work
    with (no banding). Other formats are as precise as they were stored (usually 8 bits)."""
    if is_raw(path):
        import rawpy
        with rawpy.imread(path) as r:
            half = bool(max_edge) and max(r.sizes.width, r.sizes.height) >= 2 * max_edge
            rgb = r.postprocess(half_size=half, use_camera_wb=True, output_bps=16, output_color=rawpy.ColorSpace.sRGB,
                                highlight_mode=rawpy.HighlightMode.Blend)
        return rgb.astype(np.float32) / 65535.0
    return np.asarray(load(path, max_edge), dtype=np.float32) / 255.0


def image_size(path: str) -> tuple:
    """Upright pixel size without decoding the whole image."""
    if is_raw(path):
        import rawpy
        with rawpy.imread(path) as r:
            w, h = r.sizes.width, r.sizes.height
            return (h, w) if r.sizes.flip in (5, 6) and w > h else (w, h)
    with Image.open(path) as im:
        w, h = im.size
        orient = im.getexif().get(0x0112, 1)
        return (h, w) if orient in (5, 6, 7, 8) else (w, h)


_SRGB_PROFILE = ImageCms.createProfile("sRGB")
_SRGB_BYTES = ImageCms.ImageCmsProfile(_SRGB_PROFILE).tobytes()


def to_srgb(img: Image.Image, icc: Optional[bytes]) -> tuple:
    """Convert an image with an embedded colour profile (Display P3 from iPhones, Adobe RGB, ...) to sRGB.
    All of the editor's maths (white balance, skin hue, gamma) assumes sRGB, and the browser previews are sRGB, so
    working in sRGB keeps what you see, what is analysed and what is exported identical. Returns (image, profile)."""
    if not icc:
        return img, None
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        if "srgb" in (src.profile.profile_description or "").lower():
            return img, icc
        out = ImageCms.profileToProfile(img, src, _SRGB_PROFILE, renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                                        outputMode="RGB", flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)
        return out, _SRGB_BYTES
    except Exception:
        return img, icc


def load(path: str, max_edge: Optional[int] = None) -> Image.Image:
    """Open any supported photo as upright RGB, keeping EXIF and the ICC profile.
    max_edge is a speed hint: JPEG uses DCT down-scaling and raw uses half-size decode (the result may
    still be larger than max_edge, never smaller)."""
    if is_raw(path):
        return _load_raw(path, max_edge)
    img = Image.open(path)
    if max_edge and img.format == "JPEG":
        img.draft("RGB", (max_edge, max_edge))
    img.load()
    icc = img.info.get("icc_profile")
    exif = img.getexif()
    img = ImageOps.exif_transpose(img).convert("RGB")  # honour camera rotation
    img, icc = to_srgb(img, icc)                       # wide-gamut photos -> sRGB (see to_srgb)
    exif[0x0112] = 1  # pixels are already rotated; don't let viewers rotate again
    img.info["exif"], img.info["icc_profile"] = (exif.tobytes() if len(exif) else None), icc
    return img


def _thumb(img: Image.Image, edge: int = 1024) -> np.ndarray:
    t = img.copy()
    t.thumbnail((edge, edge))
    return np.asarray(t, dtype=np.float32) / 255.0


def analyze(img: Image.Image) -> Adjustments:
    """Propose corrections from histogram statistics. Deliberately conservative."""
    a = _thumb(img)
    y = a @ LUMA
    p1, p5, p10, p50, p95, p99 = np.percentile(y, [1, 5, 10, 50, 95, 99])

    # Exposure: move the median toward a mid target, damped and clamped.
    target = 0.46
    # Asymmetric: bright skies pull the median up, so never darken much (highlights handle that).
    ev = float(np.clip(0.7 * np.log2(target**GAMMA / max(p50, 0.02) ** GAMMA), -0.15, 0.8))
    # Don't brighten into clipped highlights.
    if p99 > 0.96 and ev > 0:
        ev *= 0.3

    # Highlight / shadow recovery from the tails (measured after the exposure shift).
    shifted_p99 = min(1.0, p99 * 2 ** (ev / GAMMA))
    highlights = -float(np.clip((shifted_p99 - 0.90) * 600, 0, 60))
    shadows = float(np.clip((0.22 - p10) * 400, 0, 55))

    # Contrast: only add when the tonal range is flat.
    spread = p95 - p5
    contrast = float(np.clip((0.70 - spread) * 100, 0, 25))

    # White balance: gray-world on low-chroma midtones, damped. Skipped if there are too
    # few neutral pixels (sunsets, foliage, coloured walls) so we don't "fix" real colour.
    chroma = a.max(axis=2) - a.min(axis=2)
    mask = (y > 0.2) & (y < 0.8) & (chroma < 0.25)
    temperature = tint = 0.0
    if mask.mean() > 0.08:
        r, g, b = (float(a[..., i][mask].mean()) + 1e-4 for i in range(3))
        # Asymmetric: warm light (golden hour, skin) is usually intended, so cool shifts are small.
        temperature = float(np.clip(0.4 * 200 * np.log2(b / r), -6, 20))
        tint = float(np.clip(-0.4 * (100 / 0.15) * np.log2(np.sqrt(r * b) / g), -10, 10))

    return Adjustments(exposure=ev, temperature=temperature, tint=tint, contrast=contrast,
                       highlights=highlights, shadows=shadows)


def edge_aware_blur(y: np.ndarray, radius: float, eps: float = 0.02) -> np.ndarray:
    """Blur a single-channel float image without smearing across edges (guided filter, computed at
    reduced size). Plain Gaussian masks create halos next to dark objects against bright skies."""
    h, w = y.shape
    scale = min(1.0, 700 / max(h, w))
    small = cv2.resize(y, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else y
    r = max(2, int(radius * scale))
    box = lambda a: cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1))
    mean = box(small)
    var = box(small * small) - mean * mean
    a = var / (var + eps * eps)
    b = mean - a * mean
    out = box(a) * small + box(b)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC) if scale < 1 else out


def _blur_mask(y: np.ndarray, radius: float) -> np.ndarray:
    return np.clip(edge_aware_blur(np.clip(y, 0, 1).astype(np.float32), radius), 0, 1)


def apply_array(arr: np.ndarray, adj: Adjustments, protect: Optional[np.ndarray] = None) -> np.ndarray:
    """Tone/colour engine on a float32 RGB array in [0,1] (copied, not modified in place).

    `protect` (HxW, 0..1) shields regions such as skin from vibrance/saturation boosts.
    """
    arr = arr.copy()
    h, w = arr.shape[:2]
    long_edge = max(h, w)

    # 1. Exposure + white balance in (approximately) linear light.
    if adj.exposure or adj.temperature or adj.tint:
        gains = np.array([
            2 ** (0.25 * adj.temperature / 100),
            2 ** (-0.15 * adj.tint / 100),
            2 ** (-0.25 * adj.temperature / 100),
        ], dtype=np.float32) * np.float32(2 ** adj.exposure)
        arr = np.power(arr, GAMMA)
        arr *= gains
        np.clip(arr, 0, None, out=arr)
        arr = np.power(arr, 1 / GAMMA)
        np.clip(arr, 0, 1, out=arr)

    # 2. Highlights / shadows, masked by a blurred luminance so edges don't halo.
    if adj.highlights or adj.shadows:
        yb = _blur_mask(arr @ LUMA, long_edge / 60)
        arr += (adj.highlights / 100 * 0.30 * yb**2 + adj.shadows / 100 * 0.30 * (1 - yb) ** 2)[..., None]
        np.clip(arr, 0, 1, out=arr)

    # 3. Contrast: blend toward a smoothstep S-curve.
    if adj.contrast:
        s = arr * arr * (3 - 2 * arr)
        arr += (adj.contrast / 100) * (s - arr)
        np.clip(arr, 0, 1, out=arr)

    # 4. Vibrance (muted colours first) + saturation.
    if adj.vibrance or adj.saturation:
        gray = (arr @ LUMA)[..., None]
        chroma = (arr.max(axis=2) - arr.min(axis=2))[..., None]
        factor = (1 + adj.saturation / 100) * (1 + adj.vibrance / 100 * (1 - chroma))
        if protect is not None:
            factor = 1 + (factor - 1) * (1 - 0.85 * protect[..., None])
        arr = np.clip(gray + (arr - gray) * factor, 0, 1)

    # 5. Clarity: local contrast on midtones only.
    if adj.clarity:
        y = arr @ LUMA
        detail = y - _blur_mask(y, long_edge / 150)
        arr += (adj.clarity / 100 * 0.6 * detail * 4 * y * (1 - y))[..., None]
        np.clip(arr, 0, 1, out=arr)

    return arr


def to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")


def sharpen_image(img: Image.Image, amount: float) -> Image.Image:
    """Output sharpening, radius scaled to resolution."""
    if not amount:
        return img
    long_edge = max(img.size)
    return img.filter(ImageFilter.UnsharpMask(radius=max(0.8, long_edge / 3000),
                                              percent=int(amount * 1.5), threshold=2))


def apply(img: Image.Image, adj: Adjustments) -> Image.Image:
    arr = apply_array(np.asarray(img, dtype=np.float32) / 255.0, adj)
    return sharpen_image(to_image(arr), adj.sharpen)


def save(img: Image.Image, original: Image.Image, path: str) -> None:
    kwargs: dict = dict(quality=95, subsampling=0, optimize=True)
    if original.info.get("icc_profile"):
        kwargs["icc_profile"] = original.info["icc_profile"]
    exif: Optional[bytes] = original.info.get("exif")
    if exif:
        kwargs["exif"] = exif
    img.save(path, "JPEG", **kwargs)
