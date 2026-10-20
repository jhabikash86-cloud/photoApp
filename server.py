import hashlib
import json
import os
import re
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import adjust
import engine
import drama_looks
import exporter
import geometry
import library
import look_engine
import luts
import sky_engine
import sky_replace
import match
import immich
from engine import Analysis
from params import EditParams, MaskLayer, Stroke

MAX_UPLOAD_BYTES = 400 * 1024 * 1024   # raw files are big
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff"} | adjust.RAW_EXT
MAX_SESSIONS = 5  # sessions kept; only the 2 most recent keep their render caches in memory
KEEP_FILES_HOURS = 24
UPLOAD_DIR = Path("temp_uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def _cleanup_old_files():
    """Delete previews, exports and uploads older than KEEP_FILES_HOURS that no live session needs."""
    import time
    live = set(UPLOADS)
    cutoff = time.time() - KEEP_FILES_HOURS * 3600
    for folder in (OUTPUT_DIR, UPLOAD_DIR):
        for f in folder.iterdir():
            try:
                if f.is_file() and f.stat().st_mtime < cutoff and not any(f.name.startswith(s) for s in live):
                    f.unlink()
            except OSError:
                pass


def _cleanup_loop():
    import time
    while True:
        _cleanup_old_files()
        time.sleep(3600)


app = FastAPI()
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
sky_replace.CACHE.mkdir(parents=True, exist_ok=True)
app.mount("/sky_thumbs", StaticFiles(directory=sky_replace.CACHE), name="sky_thumbs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

UPLOADS: "OrderedDict[str, Path]" = OrderedDict()      # uploaded, waiting for / holding an analysis
SESSIONS: "OrderedDict[str, Analysis]" = OrderedDict()
LOCKS: dict = {}
REGISTRY_LOCK = threading.Lock()
threading.Thread(target=_cleanup_loop, daemon=True).start()


def _forget(sid: str):
    path = UPLOADS.pop(sid, None)
    SESSIONS.pop(sid, None)
    LOCKS.pop(sid, None)
    if path:
        path.unlink(missing_ok=True)
    for f in OUTPUT_DIR.glob(f"{sid}*"):
        f.unlink(missing_ok=True)


def _evict():
    while len(UPLOADS) > MAX_SESSIONS:
        _forget(next(iter(UPLOADS)))
    for sid in list(SESSIONS)[:-2]:        # older sessions give their render cache back (rebuilt on demand)
        SESSIONS[sid].cache.clear()


def _session(sid: str) -> Analysis:
    with REGISTRY_LOCK:
        an = SESSIONS.get(sid)
        if an is None:
            raise HTTPException(404, "This edit session has expired. Reopen the photo.")
        UPLOADS.move_to_end(sid)
        return an


def _histogram(img) -> dict:
    """64-bin RGB + luminance histogram of the rendered picture, normalised to 0..1."""
    t = img.copy()
    t.thumbnail((256, 256))
    a = np.asarray(t, np.uint8)
    out = {}
    for name, ch in (("r", a[..., 0]), ("g", a[..., 1]), ("b", a[..., 2]),
                     ("l", (a @ np.array([0.2126, 0.7152, 0.0722])).astype(np.uint8))):
        h, _ = np.histogram(ch, bins=64, range=(0, 256))
        out[name] = (h / max(1, h.max())).round(3).tolist()
    # clipping: the share of pixels that have lost their detail at either end (drives the histogram warning triangles)
    out["clip_hi"] = round(float((a.max(axis=2) >= 253).mean()), 4)
    out["clip_lo"] = round(float((a.max(axis=2) <= 2).mean()), 4)
    return out


def _write_preview(an: Analysis, params: EditParams, version: int, edge: int, draft: bool = False) -> dict:
    import time
    t0 = time.time()
    img = engine.render(an, params, edge, draft)
    # drafts are tiny and short-lived: a lighter JPEG and a separate file, so they never clash with a settled preview
    name = f"{an.sid}_draft.jpg" if draft else f"{an.sid}_preview.jpg"
    img.save(OUTPUT_DIR / name, "JPEG", quality=80 if draft else 90)
    return {"preview_url": f"/outputs/{name}?v={version}", "histogram": _histogram(img), "size": list(img.size),
            "draft": draft, "ms": int((time.time() - t0) * 1000)}


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    """Step 1: save the file and return a quick preview + thumbnail (before any analysis)."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'.")
    sid = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{sid}{ext}"
    try:
        size = 0
        with open(path, "wb") as buffer:
            while chunk := file.file.read(2 * 1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File too large (max 400 MB).")
                buffer.write(chunk)
        img = adjust.load(str(path), 1400)
        img.thumbnail((1400, 1400))
        img.save(OUTPUT_DIR / f"{sid}_quick.jpg", "JPEG", quality=88)
        img.thumbnail((240, 240))
        img.save(OUTPUT_DIR / f"{sid}_thumb.jpg", "JPEG", quality=82)
        full = adjust.image_size(str(path))
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except Exception as e:
        path.unlink(missing_ok=True)
        print("Upload exception:", repr(e))
        raise HTTPException(400, f"Could not open this file: {e}")
    with REGISTRY_LOCK:
        UPLOADS[sid] = path
        LOCKS[sid] = threading.Lock()
        _evict()
    return {"session_id": sid, "quick_url": f"/outputs/{sid}_quick.jpg", "thumb_url": f"/outputs/{sid}_thumb.jpg",
            "size": list(full), "raw": adjust.is_raw(str(path)), "megapixels": round(full[0] * full[1] / 1e6, 1)}


class AnalyzeRequest(BaseModel):
    session_id: str
    look: str = "auto"


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """Step 2: the one-off analysis (scene, masks, natural colour) and the automatic edit as slider values."""
    if req.look != "auto" and req.look not in engine.LOOKS:
        raise HTTPException(400, f"Unknown look '{req.look}'.")
    with REGISTRY_LOCK:
        path = UPLOADS.get(req.session_id)
        lock = LOCKS.get(req.session_id)
    if path is None:
        raise HTTPException(404, "This upload has expired. Reopen the photo.")
    with lock:
        try:
            an = engine.analyze(str(path), req.session_id)
            params = engine.auto_params(an, req.look)
            first = _write_preview(an, params, 0, engine.PREVIEW_EDGE)
            src, _, _ = engine._target(an, engine.PREVIEW_EDGE)
            adjust.to_image(src).save(OUTPUT_DIR / f"{an.sid}_original.jpg", "JPEG", quality=90)
        except Exception as e:
            print("Analysis exception:", repr(e))
            raise HTTPException(500, f"Pipeline error: {e}")
        with REGISTRY_LOCK:
            SESSIONS[req.session_id] = an
    return {
        "status": "success", "session_id": req.session_id, "original_url": f"/outputs/{an.sid}_original.jpg",
        "params": params.model_dump(), "neutral": EditParams().model_dump(), "looks": list(engine.LOOKS),
        "regions": [k for k in an.bank.names() if k not in ("hair", "person")], "lighting_notes": an.lighting_notes,
        "analysis": an.summary(), "full_size": list(an.full_size), **first,
    }


class AutoRequest(BaseModel):
    session_id: str
    look: str = "auto"
    light: Optional[str] = None      # override the detected light (golden_hour, sunset, blue_hour, sunny, overcast, night, indoor)
    scene: Optional[str] = None      # override the detected scene (mountain_lake, mountains, coast, forest, ...)


@app.post("/api/auto")
def auto(req: AutoRequest):
    """The automatic edit for a look, as slider values (what choosing a preset does)."""
    an = _session(req.session_id)
    if req.look != "auto" and req.look not in engine.LOOKS:
        raise HTTPException(400, f"Unknown look '{req.look}'.")
    with LOCKS[req.session_id]:
        params = engine.auto_params(an, req.look, req.light, req.scene)
        read = an.read.summary() if an.read else None
    return {"params": params.model_dump(), "lighting_notes": an.lighting_notes, "grade_notes": an.grade_notes, "read": read}


class LightingRequest(BaseModel):
    session_id: str


@app.post("/api/auto_lighting")
def auto_lighting(req: LightingRequest):
    """The automatic dodge & burn proposal (and why), without touching any other setting."""
    an = _session(req.session_id)
    with LOCKS[req.session_id]:
        db = engine.auto_lighting(an)
    return {"dodge_burn": db.model_dump(), "notes": an.lighting_notes}


class MaskOverlayRequest(BaseModel):
    session_id: str
    name: str
    mask_strength: float = 100
    feather: float = 0
    refine: float = 0
    invert: bool = False
    strokes: List[Stroke] = []
    version: int = 0


@app.post("/api/mask_overlay")
def mask_overlay(req: MaskOverlayRequest):
    """Overlay of ONE mask with the current feather / refine / invert / strength, for the Selective Masking panel."""
    an = _session(req.session_id)
    with LOCKS[req.session_id]:
        try:
            img = engine.mask_overlay(an, req.name, req.mask_strength, req.feather, req.refine, req.invert, strokes=req.strokes)
        except ValueError as e:
            raise HTTPException(404, str(e))
        img.save(OUTPUT_DIR / f"{an.sid}_mask.jpg", "JPEG", quality=85)
    return {"url": f"/outputs/{an.sid}_mask.jpg?v={req.version}"}


class RenderRequest(BaseModel):
    session_id: str
    params: EditParams
    version: int = 0
    edge: Optional[int] = None
    draft: bool = False


@app.post("/api/render")
def render(req: RenderRequest):
    """Re-render from the cached analysis (fast): used for every slider move."""
    an = _session(req.session_id)
    with LOCKS[req.session_id]:
        try:
            return _write_preview(an, req.params, req.version, req.edge or engine.PREVIEW_EDGE, req.draft)
        except ValueError as e:
            raise HTTPException(400, str(e))


class ExportRequest(BaseModel):
    session_id: str
    params: EditParams
    version: int = 0
    quality: int = 95


@app.post("/api/export")
def export(req: ExportRequest):
    """Full-resolution render with the same sliders, saved as a JPEG with the original EXIF/profile."""
    an = _session(req.session_id)
    with LOCKS[req.session_id]:
        img = engine.render(an, req.params, None)
        name = f"{an.sid}_full_{req.version}.jpg"
        original = adjust.load(an.path, 64)  # only for EXIF / colour profile
        original.info.setdefault("icc_profile", None)
        adjust.save(img, original, str(OUTPUT_DIR / name))
    return {"url": f"/outputs/{name}", "size": list(img.size)}


class ExportTo(BaseModel):
    session_id: str
    params: EditParams
    preset: str = "full"
    name: str = "photo"
    version: int = 0
    max_edge: Optional[int] = None
    quality: Optional[int] = None


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "photo"
    return stem[:80]


@app.get("/api/export_presets")
def export_presets():
    return {"presets": [{"id": k, "label": v["label"], "format": v["format"], "max_edge": v["max_edge"]} for k, v in exporter.PRESETS.items()]}


@app.post("/api/export_to")
def export_to(req: ExportTo):
    """One photo, one preset (web, social, print TIFF, ...)."""
    if req.preset not in exporter.PRESETS:
        raise HTTPException(400, "Unknown export preset.")
    an = _session(req.session_id)
    with LOCKS[req.session_id]:
        path, size = exporter.export(an, req.params, req.preset, str(OUTPUT_DIR / f"{an.sid}_{req.preset}_{req.version}"), req.max_edge, req.quality)
    return {"url": f"/outputs/{Path(path).name}", "size": list(size), "filename": f"{_safe_stem(req.name)}-{req.preset}{Path(path).suffix}"}


class ExportItem(BaseModel):
    session_id: str
    params: EditParams
    name: str


class ExportBatch(BaseModel):
    items: List[ExportItem]
    preset: str = "web"


EXPORT_JOBS: "OrderedDict[str, dict]" = OrderedDict()


def _run_export_job(job_id: str, req: ExportBatch):
    import zipfile
    job = EXPORT_JOBS[job_id]
    out = OUTPUT_DIR / f"export_{job_id}.zip"
    used: set = set()
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
            for n, it in enumerate(req.items, 1):
                if job.get("cancel"):
                    job["status"] = "cancelled"
                    return
                job["current"] = it.name
                try:
                    an = _session(it.session_id)
                    with LOCKS[it.session_id]:
                        path, _ = exporter.export(an, it.params, req.preset, str(OUTPUT_DIR / f"{job_id}_{n}"))
                    stem, ext = _safe_stem(it.name), Path(path).suffix
                    arc, k = f"{stem}-{req.preset}{ext}", 1
                    while arc in used:
                        k += 1
                        arc = f"{stem}-{req.preset}-{k}{ext}"
                    used.add(arc)
                    z.write(path, arc)
                    Path(path).unlink(missing_ok=True)
                except Exception as e:                       # one bad photo must not stop the rest
                    job["failed"] += 1
                    job["errors"].append(f"{it.name}: {str(e)[:120]}")
                job["done"] = n
        job["url"] = f"/outputs/{out.name}"
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["errors"].append(str(e)[:200])


@app.post("/api/export_batch")
def export_batch(req: ExportBatch):
    """Every photo with its own edit, delivered as one ZIP (runs in the background; poll /api/export_jobs/{id})."""
    if req.preset not in exporter.PRESETS:
        raise HTTPException(400, "Unknown export preset.")
    if not req.items:
        raise HTTPException(400, "Nothing to export.")
    job_id = uuid.uuid4().hex[:12]
    EXPORT_JOBS[job_id] = dict(id=job_id, status="running", done=0, total=len(req.items), failed=0, current="", errors=[], url=None, preset=req.preset)
    while len(EXPORT_JOBS) > 20:
        EXPORT_JOBS.popitem(last=False)
    threading.Thread(target=_run_export_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/export_jobs/{job_id}")
def export_job(job_id: str):
    job = EXPORT_JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown export job.")
    return {k: v for k, v in job.items() if k != "cancel"}


@app.post("/api/export_jobs/{job_id}/cancel")
def export_cancel(job_id: str):
    job = EXPORT_JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown export job.")
    job["cancel"] = True
    return {"ok": True}


@app.get("/api/dust/{sid}")
def dust_count(sid: str):
    """How many sensor-dust spots were found in the sky (to offer the clean-up)."""
    import spots
    an = _session(sid)
    with LOCKS[sid]:
        src, size, _ = engine._target(an, 1400)
        m = engine._masks_at(an, size)
        if "sky" not in m:
            return {"count": 0}
        return {"count": len(spots.detect(src, m["sky"], 60.0))}


@app.get("/api/geometry/{sid}")
def geometry_suggest(sid: str):
    """How much to rotate to level the photo (from the horizon, else from strong straight lines)."""
    an = _session(sid)
    with LOCKS[sid]:
        src, size, _ = engine._target(an, 900)
        return geometry.auto_angle(src, dict(engine._masks_at(an, size)))


@app.get("/api/regions/{sid}")
def regions(sid: str):
    an = _session(sid)
    with LOCKS[sid]:
        engine.regions_overlay(an).save(OUTPUT_DIR / f"{sid}_regions.jpg", "JPEG", quality=85)
    return {"url": f"/outputs/{sid}_regions.jpg"}


# ---------------------------------------------------------------------------------------------
# Immich: batch edit a whole album (credentials stay on the server, in .env)
# ---------------------------------------------------------------------------------------------
@app.get("/api/immich/albums")
def immich_albums():
    if not immich.configured():
        return {"configured": False, "albums": []}
    try:
        return {"configured": True, "albums": immich.list_albums()}
    except Exception as e:
        raise HTTPException(502, f"Could not reach Immich: {e}")


class BatchRequest(BaseModel):
    album_id: str
    look: str = "auto"
    limit: Optional[int] = None
    straighten: bool = False


@app.post("/api/immich/batch")
def immich_batch(req: BatchRequest):
    if not immich.configured():
        raise HTTPException(400, "Immich is not configured (set IMMICH_URL and IMMICH_API_KEY in .env).")
    if req.look not in ("auto", "drama_auto") and req.look not in engine.LOOKS:
        raise HTTPException(400, f"Unknown look '{req.look}'.")
    try:
        return {"job_id": immich.start_batch(req.album_id, req.look, req.limit, req.straighten)}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Could not reach Immich: {e}")


@app.get("/api/immich/jobs/{job_id}")
def immich_job(job_id: str):
    job = immich.JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return {k: v for k, v in job.items() if k != "cancel"}


@app.post("/api/immich/jobs/{job_id}/cancel")
def immich_cancel(job_id: str):
    job = immich.JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    job["cancel"] = True
    return {"ok": True}


class CustomOverlayRequest(BaseModel):
    session_id: str
    layer: MaskLayer
    version: int = 0


@app.post("/api/custom_overlay")
def custom_overlay(req: CustomOverlayRequest):
    """Overlay of ONE custom mask (gradient, radial, brush, luminosity, colour) exactly as it is currently defined."""
    an = _session(req.session_id)
    with LOCKS[req.session_id]:
        try:
            img = engine.custom_overlay(an, req.layer)
        except ValueError as e:
            raise HTTPException(400, str(e))
        img.save(OUTPUT_DIR / f"{an.sid}_cmask.jpg", "JPEG", quality=80)
    return {"url": f"/outputs/{an.sid}_cmask.jpg?v={req.version}"}


# ---------------------------------------------------------------------------------------------
# LUTs (.cube) and saved presets
# ---------------------------------------------------------------------------------------------
@app.get("/api/luts")
def luts_list():
    return {"luts": luts.list_luts()}


@app.post("/api/luts/upload")
def luts_upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".cube"):
        raise HTTPException(400, "Choose a .cube LUT file.")
    data = file.file.read(luts.MAX_CUBE_BYTES + 1)
    try:
        lut_id = luts.save_user(file.filename, data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": lut_id, "luts": luts.list_luts()}


class LutDelete(BaseModel):
    id: str


@app.post("/api/luts/delete")
def luts_delete(req: LutDelete):
    if not luts.delete_user(req.id):
        raise HTTPException(404, "That LUT cannot be deleted.")
    return {"luts": luts.list_luts()}


class LutPreviewRequest(BaseModel):
    session_id: str
    params: EditParams


@app.post("/api/lut_previews")
def lut_previews(req: LutPreviewRequest):
    """A small preview of the current edit under every LUT (the LUT itself is switched off for the base render)."""
    import base64, io
    an = _session(req.session_id)
    p = req.params.model_copy(deep=True)
    p.lut.id = None
    with LOCKS[req.session_id]:
        base = np.asarray(engine.render(an, p, 240, True), np.float32) / 255.0
    out = {}
    for item in luts.list_luts():
        lut = luts.get(item["id"])
        if lut is None:
            continue
        img = adjust.Image.fromarray((luts.apply(base, lut, 100.0) * 255 + 0.5).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=78)
        out[item["id"]] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return {"previews": out}


PRESET_DIR = Path("presets")
PRESET_DIR.mkdir(exist_ok=True)


def _preset_path(pid: str) -> Path:
    import re
    if not re.fullmatch(r"[a-f0-9]{8,32}", pid):
        raise HTTPException(400, "Bad preset id.")
    return PRESET_DIR / f"{pid}.json"


class PresetSave(BaseModel):
    name: str
    scopes: List[str] = []
    params: dict


@app.get("/api/presets")
def presets_list():
    import json
    items = []
    for f in sorted(PRESET_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            d = json.loads(f.read_text())
            items.append({"id": f.stem, "name": d.get("name", f.stem), "scopes": d.get("scopes", []), "params": d.get("params", {})})
        except Exception:
            continue
    return {"presets": items}


@app.post("/api/presets")
def presets_save(req: PresetSave):
    import json
    name = req.name.strip()[:60]
    if not name:
        raise HTTPException(400, "Give the preset a name.")
    pid = uuid.uuid4().hex[:12]
    _preset_path(pid).write_text(json.dumps({"name": name, "scopes": req.scopes, "params": req.params}))
    return presets_list()


class PresetDelete(BaseModel):
    id: str


@app.post("/api/presets/delete")
def presets_delete(req: PresetDelete):
    path = _preset_path(req.id)
    if path.exists():
        path.unlink()
    return presets_list()


# ---------------------------------------------------------------------------------------------
# Match a reference
# ---------------------------------------------------------------------------------------------
REFS: "OrderedDict[str, tuple]" = OrderedDict()         # ref_id -> (Look, thumb url); the reference file itself is not kept


@app.post("/api/reference")
def reference_upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'.")
    rid = uuid.uuid4().hex[:12]
    path = UPLOAD_DIR / f"ref_{rid}{ext}"
    try:
        size = 0
        with open(path, "wb") as buffer:
            while chunk := file.file.read(2 * 1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File too large (max 400 MB).")
                buffer.write(chunk)
        arr = adjust.load_float(str(path), 800)
        look = match.measure(arr)
        thumb = adjust.load(str(path), 240)
        thumb.thumbnail((240, 240))
        thumb.save(OUTPUT_DIR / f"ref_{rid}.jpg", "JPEG", quality=82)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read this reference: {e}")
    finally:
        path.unlink(missing_ok=True)
    REFS[rid] = (look, f"/outputs/ref_{rid}.jpg")
    while len(REFS) > 12:
        REFS.popitem(last=False)
    return {"ref_id": rid, "thumb_url": f"/outputs/ref_{rid}.jpg"}


class MatchRequest(BaseModel):
    session_id: str
    ref_id: str
    params: EditParams           # the sliders BEFORE matching (the client keeps this snapshot)
    strength: float = 70


@app.post("/api/match")
def match_reference(req: MatchRequest):
    an = _session(req.session_id)
    if req.ref_id not in REFS:
        raise HTTPException(404, "That reference is no longer available. Upload it again.")
    look = REFS[req.ref_id][0]
    key = f"match:{req.ref_id}:{hash(req.params.model_dump_json())}"
    with LOCKS[req.session_id]:
        m = an.cache.get(key)
        if m is None:
            def rs(p):
                return np.asarray(engine.render(an, p, 420, True), np.float32) / 255.0
            m = match.solve(rs, req.params, look)
            an.cache[key] = m
    out = match.blend(req.params, m.full, req.strength / 100.0)
    return {"params": out.model_dump(), "notes": m.notes}


# ---------------------------------------------------------------------------------------------
# Scene-aware looks
# ---------------------------------------------------------------------------------------------
@app.get("/api/looks")
def looks_list():
    return {"looks": look_engine.list_looks()}


class LookRequest(BaseModel):
    session_id: str
    look_id: str
    params: EditParams            # the current sliders (white balance, people, masks and effects are kept)
    strength: float = 100         # percent, 0..150
    light: Optional[str] = None
    scene: Optional[str] = None
    smart_sky: bool = False       # also apply the best-fitting premium sky treatment


@app.post("/api/look")
def apply_look(req: LookRequest):
    an = _session(req.session_id)
    look = look_engine.LOOKS.get(req.look_id)
    if look is None:
        raise HTTPException(404, "Unknown look.")
    with LOCKS[req.session_id]:
        p, notes, facts = look_engine.build(an, req.params, look, req.strength / 100.0, req.light, req.scene)
        if req.smart_sky:
            r = sky_engine.read_sky(an)
            if r is not None and r.share >= 0.08:
                fits = sky_engine.fit_scores(r, facts.get("light"))
                best = max(fits, key=fits.get)
                if fits[best] >= 0.5:
                    p, sky_notes, _ = sky_engine.apply(an, p, best, 0.8 * req.strength / 100.0)
                    notes += [f"Smart sky: {sky_engine.TREATMENTS[best].label}"] + sky_notes[1:]
                    facts["sky_treatment"] = best
    return {"params": p.model_dump(), "notes": notes, "facts": facts}


# ---------------------------------------------------------------------------------------------
# Sky studio and suggestions
# ---------------------------------------------------------------------------------------------
@app.get("/api/suggest/{sid}")
def suggest(sid: str):
    an = _session(sid)
    with LOCKS[sid]:
        return sky_engine.suggest(an)


class SkyRequest(BaseModel):
    session_id: str
    treatment: str
    params: EditParams
    strength: float = 100


@app.post("/api/sky")
def apply_sky(req: SkyRequest):
    an = _session(req.session_id)
    with LOCKS[req.session_id]:
        try:
            p, notes, sky = sky_engine.apply(an, req.params, req.treatment, req.strength / 100.0)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"params": p.model_dump(), "notes": notes, "sky": sky}


# ---------------------------------------------------------------------------------------------
# Sky replacement (skies from the local CC0 library)
# ---------------------------------------------------------------------------------------------
@app.get("/api/skies/{sid}")
def skies_for(sid: str):
    """Is a new sky worth it for this photo, and the library skies ranked by how well they suit its light."""
    an = _session(sid)
    import look_engine
    with LOCKS[sid]:
        info = sky_replace.need(an)
        if not sky_replace.library():
            return {**info, "available": False, "reason": "The sky library is not downloaded (run fetch_skies.py).", "skies": []}
        facts = look_engine.read_facts(an, None, None)
        ranked = sky_replace.rank(an, facts.light, look_engine.measure_tonal(an))
    for r in ranked:
        sky_replace.thumb_path(r["id"])
        r["thumb"] = f"/sky_thumbs/{r['id']}_thumb.jpg"
    return {**info, "light": facts.light, "skies": ranked}


class SkyPlanRequest(BaseModel):
    session_id: str
    sky_id: str


@app.post("/api/sky_plan")
def sky_plan(req: SkyPlanRequest):
    """Make the adaptive decisions for putting this library sky into this photo (composition, brightness, haze, relight)."""
    an = _session(req.session_id)
    with LOCKS[req.session_id]:
        try:
            sr = sky_replace.plan(an, req.sky_id)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"sky_replace": sr.model_dump()}


# ---------------------------------------------------------------------------------------------
# Library: read-only catalog of the travel photos on the external drive (see library.py / build_library.py)
# ---------------------------------------------------------------------------------------------
@app.get("/api/library/albums")
def library_albums():
    return library.albums()


@app.get("/api/library/photos")
def library_photos(album: str = "all", order: str = "taken", limit: int = 120, offset: int = 0, kind: Optional[str] = None, subject: Optional[str] = None):
    try:
        return library.photos(album, order, min(max(limit, 1), 400), max(offset, 0), kind, subject)
    except ValueError:
        raise HTTPException(400, "Unknown album.")


@app.get("/api/library/thumb/{pid}")
def library_thumb(pid: int):
    f = library.THUMB_DIR / f"{pid}.jpg"
    if f.exists():
        return FileResponse(f, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})
    ph = library.photo(pid)
    if ph is None:
        raise HTTPException(404, "No such photo.")
    # a duplicate or an edited export shares the thumbnail of the photo it came from; otherwise make one from the drive on demand
    for other in (ph.get("dup_of"), ph.get("raw_id")):
        if other and (library.THUMB_DIR / f"{other}.jpg").exists():
            return FileResponse(library.THUMB_DIR / f"{other}.jpg", media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})
    path = library.abs_path(ph["path"])
    if not os.path.exists(path):
        raise HTTPException(404, "The drive is not connected.")
    import build_library
    try:
        im, _ = build_library._preview(path)
        im.thumbnail((480, 480))
        im.save(f, "JPEG", quality=82)
    except Exception as e:
        raise HTTPException(415, f"Could not read this file: {e}")
    return FileResponse(f, media_type="image/jpeg")


def _immich_asset_for(pid: int) -> Optional[str]:
    """The Immich copy of a library photo (or of the photo it duplicates), if it has been uploaded."""
    db = library.connect()
    try:
        row = db.execute("SELECT m.asset_id FROM immich_map m JOIN photos p ON m.photo_id=COALESCE(p.dup_of, p.id) WHERE p.id=?", (pid,)).fetchone()
    except Exception:
        return None
    return row[0] if row else None


@app.get("/api/library/file/{pid}")
def library_file(pid: int):
    """The original file. From the external drive when it is plugged in; otherwise streamed from the copy in Immich."""
    from fastapi.responses import StreamingResponse
    ph = library.photo(pid)
    if ph is None:
        raise HTTPException(404, "No such photo.")
    path = library.abs_path(ph["path"])
    if os.path.exists(path):
        return FileResponse(path, filename=os.path.basename(path))
    aid = _immich_asset_for(pid)
    if aid and immich.configured():
        import requests
        base = (os.environ.get("IMMICH_UPLOAD_URL") or os.environ["IMMICH_URL"]).rstrip("/")
        up = requests.get(f"{base}/api/assets/{aid}/original", headers={**immich.HEADERS_BASE, "x-api-key": os.environ["IMMICH_API_KEY"]}, stream=True, timeout=120)
        if up.status_code == 200:
            return StreamingResponse(up.iter_content(1 << 20), media_type="application/octet-stream",
                                     headers={"Content-Disposition": f'attachment; filename="{os.path.basename(ph["path"])}"', "Content-Length": up.headers.get("Content-Length", "")})
    raise HTTPException(404, "The external drive is not connected, and this photo is not in Immich yet.")


# ---------------------------------------------------------------------------------------------
# Autosaved edits: every photo reopens exactly as it was left (survives a refresh or a server restart)
# ---------------------------------------------------------------------------------------------
EDIT_DIR = Path("edits")
EDIT_DIR.mkdir(exist_ok=True)


def _edit_file(name: str, size: int) -> Path:
    key = hashlib.sha1(f"{name}|{size}".encode()).hexdigest()[:24]
    return EDIT_DIR / f"{key}.json"


@app.get("/api/edit")
def edit_load(name: str, size: int):
    f = _edit_file(name, size)
    if not f.exists():
        raise HTTPException(404, "No saved edit.")
    try:
        d = json.loads(f.read_text())
        d["params"] = EditParams(**d["params"]).model_dump()    # validated AND completed: fields added since it was saved get their defaults
    except Exception:
        raise HTTPException(404, "No usable saved edit.")
    return d


class EditSave(BaseModel):
    name: str
    size: int
    params: EditParams


@app.post("/api/edit")
def edit_save(req: EditSave):
    import time
    f = _edit_file(req.name, req.size)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"name": req.name, "size": req.size, "saved": time.time(), "params": req.params.model_dump()}))
    os.replace(tmp, f)
    return {"ok": True}


class EditKey(BaseModel):
    name: str
    size: int


@app.post("/api/edit/delete")
def edit_delete(req: EditKey):
    _edit_file(req.name, req.size).unlink(missing_ok=True)
    return {"ok": True}


# ---------------------------------------------------------------------------------------------
# Drama looks (generated for each photo)
# ---------------------------------------------------------------------------------------------
DRAMA_THUMBS: "OrderedDict[tuple, bytes]" = OrderedDict()
DRAMA_BUILDING: set = set()


def _prepare_drama_thumbs(sid: str, an, order):
    """Build the preview of every generated look, one after another (parallel builds just queue on the photo's lock)."""
    import io
    try:
        base = engine.auto_params(an, "auto")
        for vid in order:
            if (sid, vid) in DRAMA_THUMBS:
                continue
            try:
                with LOCKS[sid]:
                    p, _, _ = drama_looks.build(an, base, vid, 1.0)
                    img = engine.render(an, p, 300, True)
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=80)
                DRAMA_THUMBS[(sid, vid)] = buf.getvalue()
            except Exception as e:
                print("drama thumb failed", vid, repr(e)[:100])
                DRAMA_THUMBS[(sid, vid)] = b""
            while len(DRAMA_THUMBS) > 300:
                DRAMA_THUMBS.popitem(last=False)
    finally:
        DRAMA_BUILDING.discard(sid)


@app.get("/api/drama/{sid}")
def drama_list(sid: str):
    """The Drama looks that suit THIS photo, ranked (scene analysed first). Previews are built in the background."""
    an = _session(sid)
    with LOCKS[sid]:
        looks = drama_looks.generate(an)
        s = drama_looks.read_scene(an)
    if sid not in DRAMA_BUILDING and any((sid, l["id"]) not in DRAMA_THUMBS for l in looks):
        DRAMA_BUILDING.add(sid)
        threading.Thread(target=_prepare_drama_thumbs, args=(sid, an, [l["id"] for l in looks]), daemon=True).start()
    return {"looks": looks, "scene": {"light": s.light, "sky": s.sky_kind, "grey": round(s.neutral, 2), "cast_strength": round(s.cast, 1),
                                      "bright_light": round(s.bright_source, 2), "horizon": s.has_horizon, "people": s.people}}


@app.get("/api/drama_thumb/{sid}/{vid}")
def drama_thumb(sid: str, vid: str, v: int = 0):
    import time
    from fastapi.responses import Response
    _session(sid)
    if vid not in drama_looks.VARIANTS:
        raise HTTPException(404, "Unknown look.")
    for _ in range(240):                                   # up to ~2 minutes: the previews are being built one by one
        data = DRAMA_THUMBS.get((sid, vid))
        if data is not None:
            break
        time.sleep(0.5)
    if not data:
        raise HTTPException(404, "Preview not available.")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


class DramaRequest(BaseModel):
    session_id: str
    look_id: str
    params: EditParams
    strength: float = 100


@app.post("/api/drama")
def drama_apply(req: DramaRequest):
    an = _session(req.session_id)
    if req.look_id not in drama_looks.VARIANTS:
        raise HTTPException(404, "Unknown look.")
    with LOCKS[req.session_id]:
        p, notes, facts = drama_looks.build(an, req.params, req.look_id, req.strength / 100.0)
    return {"params": p.model_dump(), "notes": notes, "facts": facts}
