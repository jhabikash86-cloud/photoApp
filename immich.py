"""Immich integration and batch editing.

Credentials come from .env (IMMICH_URL, IMMICH_API_KEY) and never leave the server. A batch job takes an
album, edits every photo with the automatic edit for the chosen look at full resolution, uploads each
result as a NEW photo (originals are never touched) and collects them in a new "<album> - Edited" album.
"""
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

import adjust
import engine

load_dotenv()
HEADERS_BASE = {"Accept": "application/json", "User-Agent": "curl/8.7.1"}  # Cloudflare blocks python-urllib


def configured() -> bool:
    return bool(os.environ.get("IMMICH_URL") and os.environ.get("IMMICH_API_KEY"))


def _base() -> str:
    return os.environ["IMMICH_URL"].rstrip("/")


def call(method: str, path: str, body=None, raw: bool = False, timeout: int = 120):
    headers = dict(HEADERS_BASE, **{"x-api-key": os.environ["IMMICH_API_KEY"]})
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    req = urllib.request.Request(_base() + path, method=method, headers=headers, data=data)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read()
            return r.status, (payload if raw else json.loads(payload or b"null"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:300]


def upload_image(path: str, filename: str, created: str, modified: str):
    boundary = uuid.uuid4().hex
    parts: List[bytes] = []
    for k, v in (("deviceAssetId", f"neural-edit-{uuid.uuid4().hex}"), ("deviceId", "neural-edit-pro"),
                 ("fileCreatedAt", created), ("fileModifiedAt", modified)):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="assetData"; filename="{filename}"\r\n'
                 f'Content-Type: image/jpeg\r\n\r\n'.encode())
    parts.append(Path(path).read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    headers = dict(HEADERS_BASE, **{"x-api-key": os.environ["IMMICH_API_KEY"],
                                    "Content-Type": f"multipart/form-data; boundary={boundary}"})
    req = urllib.request.Request(_base() + "/api/assets", method="POST", headers=headers, data=b"".join(parts))
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:300]


def list_albums() -> List[dict]:
    st, albums = call("GET", "/api/albums")
    if st != 200:
        raise RuntimeError(f"Immich said {st}: {albums}")
    return sorted([{"id": a["id"], "name": a["albumName"], "count": a["assetCount"]} for a in albums if a["assetCount"] > 0],
                  key=lambda a: a["name"].lower())


def album_images(album_id: str) -> List[dict]:
    out, page = [], 1
    while page:
        st, r = call("POST", "/api/search/metadata", {"albumIds": [album_id], "size": 100, "page": page, "type": "IMAGE"})
        if st != 200:
            raise RuntimeError(f"Immich said {st}: {r}")
        out += r["assets"]["items"]
        page = int(r["assets"]["nextPage"]) if r["assets"].get("nextPage") else 0
    return out


# ---------------------------------------------------------------------------------------------
# batch jobs
# ---------------------------------------------------------------------------------------------
JOBS: Dict[str, dict] = {}
_job_lock = threading.Lock()


def start_batch(album_id: str, look: str, limit: Optional[int] = None) -> str:
    albums = {a["id"]: a for a in list_albums()}
    if album_id not in albums:
        raise ValueError("Unknown album")
    job_id = uuid.uuid4().hex[:12]
    job = dict(id=job_id, album=albums[album_id]["name"], look=look, status="starting", done=0, total=0, failed=0,
               current="", errors=[], edited_album_id=None, cancel=False, started=time.time())
    with _job_lock:
        JOBS[job_id] = job
    threading.Thread(target=_run, args=(job, album_id, look, limit), daemon=True).start()
    return job_id


def _run(job: dict, album_id: str, look: str, limit: Optional[int]):
    try:
        assets = [a for a in album_images(album_id) if "-edited" not in a.get("originalFileName", "").lower()]
        if limit:
            assets = assets[:limit]
        job.update(total=len(assets), status="running")
        edited_ids: List[str] = []
        tmp = Path(tempfile.mkdtemp(prefix="neural_batch_"))
        for a in assets:
            if job["cancel"]:
                break
            name = a.get("originalFileName", a["id"])
            job["current"] = name
            src_path = tmp / f"{a['id']}{Path(name).suffix.lower() or '.jpg'}"
            out_path = tmp / f"{a['id']}_edit.jpg"
            try:
                st, data = call("GET", f"/api/assets/{a['id']}/original", raw=True, timeout=300)
                if st != 200:
                    raise RuntimeError(f"download failed ({st})")
                src_path.write_bytes(data)
                an = engine.analyze(str(src_path))
                params = engine.auto_params(an, look)
                img = engine.render(an, params, None)
                adjust.save(img, adjust.load(str(src_path), 64), str(out_path))
                st, up = upload_image(str(out_path), f"{Path(name).stem}-edited.jpg",
                                      a.get("fileCreatedAt", "2024-01-01T00:00:00.000Z"),
                                      a.get("fileModifiedAt", a.get("fileCreatedAt", "2024-01-01T00:00:00.000Z")))
                if st not in (200, 201):
                    raise RuntimeError(f"upload failed ({st}): {up}")
                edited_ids.append(up["id"])
            except Exception as e:
                job["failed"] += 1
                job["errors"].append(f"{name}: {e}")
            finally:
                src_path.unlink(missing_ok=True)
                out_path.unlink(missing_ok=True)
                job["done"] += 1
        if edited_ids:
            name = f"{job['album']} - Edited"
            st, alb = call("POST", "/api/albums", {"albumName": name, "description": f"Edited by Neural Edit Pro ({look})",
                                                   "assetIds": edited_ids})
            if st in (200, 201):
                job["edited_album_id"] = alb["id"]
            else:
                job["errors"].append(f"could not create album: {st} {alb}")
        job["status"] = "cancelled" if job["cancel"] else "done"
    except Exception as e:  # never let a thread die silently
        job["status"] = "error"
        job["errors"].append(str(e))
