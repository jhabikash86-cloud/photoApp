"""Put the travel library into Immich: find or upload every photo, then build the albums.

    ./venv/bin/python immich_push.py match            # find which catalog photos are already in Immich (by path, or name + size)
    ./venv/bin/python immich_push.py upload [N]       # upload the photos Immich does not have yet, from the NAS copy (streamed); N = limit
    ./venv/bin/python immich_push.py albums [--dry]   # create / fill the "Travel · ..." albums from the matches
    ./venv/bin/python immich_push.py status

Two ways to get the photos into Immich:
  * External Library (best, no upload): add the NAS copy folder as an External Library in Immich, scan it, then run `match`.
  * Upload through the API: set IMMICH_UPLOAD_URL in .env to Immich's address on your home network (http://<server-ip>:2283) and run `upload`.
Only albums starting with "Travel ·" are ever created or changed. Nothing in Immich is deleted. Originals are only read.
"""
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.parse
import uuid
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
import immich  # noqa: E402
import library as lib  # noqa: E402

PREFIX = "Travel · "
MIN_PLACE = 25                    # place albums only for places with at least this many photos


def _db():
    db = lib.connect()
    db.execute("CREATE TABLE IF NOT EXISTS immich_map (photo_id INTEGER PRIMARY KEY, asset_id TEXT, how TEXT)")
    db.commit()
    return db


# ------------------------------------------------------------------------------------------------ matching
def _all_assets():
    """Every asset in Immich: id, original file name, size, original path (paged)."""
    out, page = [], 1
    while True:
        s, r = immich.call("POST", "/api/search/metadata", {"page": page, "size": 1000, "withExif": True}, timeout=300)
        if s != 200:
            sys.exit(f"Immich search failed: {s} {str(r)[:120]}")
        items = r["assets"]["items"]
        for a in items:
            out.append({"id": a["id"], "name": a.get("originalFileName", ""), "path": a.get("originalPath", ""),
                        "size": (a.get("exifInfo") or {}).get("fileSizeInByte")})
        print(f"  read {len(out)} assets from Immich", flush=True)
        if not r["assets"].get("nextPage"):
            break
        page += 1
    return out


def _tail(path: str, n: int = 3) -> str:
    return "/".join(path.replace("\\", "/").split("/")[-n:]).lower()


def match():
    """Link each catalog photo to its Immich asset. External Library assets are recognised by their folder path (last 3 parts),
    uploaded ones by file name + size."""
    db = _db()
    assets = _all_assets()
    by_tail = {_tail(a["path"]): a["id"] for a in assets if a["path"]}
    by_ns = defaultdict(list)
    for a in assets:
        by_ns[(a["name"].lower(), a["size"])].append(a["id"])
    n_path = n_ns = 0
    for r in db.execute("SELECT id, path, size FROM photos WHERE dup_of IS NULL").fetchall():
        aid = by_tail.get(_tail(r["path"]))
        how = "path"
        if aid is None:
            ids = by_ns.get((os.path.basename(r["path"]).lower(), r["size"]))
            aid, how = (ids[0], "name+size") if ids else (None, None)
        if aid:
            db.execute("INSERT OR REPLACE INTO immich_map VALUES(?,?,?)", (r["id"], aid, how))
            n_path += how == "path"
            n_ns += how == "name+size"
    db.commit()
    print(f"matched {n_path} by path and {n_ns} by name+size (of {db.execute('SELECT COUNT(*) FROM photos WHERE dup_of IS NULL').fetchone()[0]} unique photos)", flush=True)


# ------------------------------------------------------------------------------------------------ upload
def _base():
    return os.environ.get("IMMICH_UPLOAD_URL") or os.environ["IMMICH_URL"]


def _upload_one(path: str, device_asset_id: str, created: str, modified: str, sha1: str):
    u = urllib.parse.urlparse(_base())
    Conn = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    boundary = uuid.uuid4().hex
    fields = [("deviceAssetId", device_asset_id), ("deviceId", "neural-edit-pro-travel"), ("fileCreatedAt", created), ("fileModifiedAt", modified)]
    head = b"".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields)
    head += f'--{boundary}\r\nContent-Disposition: form-data; name="assetData"; filename="{os.path.basename(path)}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    size = os.path.getsize(path)
    c = Conn(u.netloc, timeout=900)
    c.putrequest("POST", (u.path.rstrip("/") + "/api/assets"))
    for k, v in immich.HEADERS_BASE.items():
        c.putheader(k, v)
    c.putheader("x-api-key", os.environ["IMMICH_API_KEY"])
    c.putheader("x-immich-checksum", sha1)
    c.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
    c.putheader("Content-Length", str(len(head) + size + len(tail)))
    c.endheaders()
    c.send(head)
    with open(path, "rb") as f:
        while True:
            b = f.read(4 * 1024 * 1024)
            if not b:
                break
            c.send(b)
    c.send(tail)
    r = c.getresponse()
    body = r.read()
    c.close()
    try:
        return r.status, json.loads(body)
    except Exception:
        return r.status, body[:200]


def _sha1(p: str) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _immich_up() -> bool:
    try:
        s, r = immich.call("GET", "/api/server/ping", timeout=8)
        return s == 200
    except Exception:
        return False


def _wait(cond, what: str, patience_s: int = 4 * 3600) -> bool:
    """Wait (politely) for something to come back: an unplugged drive, a sleeping network, a restarting server."""
    t0, announced = time.time(), False
    while not cond():
        if not announced:
            print(f"  waiting for {what} ...", flush=True)
            announced = True
        if time.time() - t0 > patience_s:
            return False
        time.sleep(30)
    if announced:
        print(f"  {what} is back after {(time.time() - t0) / 60:.0f} min", flush=True)
    return True


def upload(limit=None):
    db = _db()
    todo = db.execute("SELECT p.id, p.path, p.taken, p.size FROM photos p WHERE p.dup_of IS NULL AND p.id NOT IN (SELECT photo_id FROM immich_map) ORDER BY p.path").fetchall()
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} photos to upload to {_base()}", flush=True)
    t0, sent, failed = time.time(), 0.0, 0
    for n, r in enumerate(todo, 1):
        src = os.path.join(lib.root(), r["path"])
        if not os.path.exists(src):
            if not _wait(lambda: os.path.exists(src), "the external drive"):
                print("  giving up: the drive did not come back", flush=True)
                break
        ts = (r["taken"] or "2000-01-01 00:00:00").replace(" ", "T")[:19] + ".000Z"
        done = False
        for attempt in range(6):
            try:
                status, body = _upload_one(src, f"neural-travel-{r['id']}", ts, ts, _sha1(src))
                if status in (200, 201) and isinstance(body, dict) and body.get("id"):
                    db.execute("INSERT OR REPLACE INTO immich_map VALUES(?,?,?)", (r["id"], body["id"], "duplicate" if body.get("status") == "duplicate" else "uploaded"))
                    done = True
                    break
                print("  REJECTED", r["path"], status, str(body)[:100], flush=True)
                if status in (400, 401, 403, 413, 415):
                    break                                   # retrying will not help
            except (OSError, http.client.HTTPException) as e:
                print(f"  retry {attempt + 1} for {r['path']}: {type(e).__name__}", flush=True)
                if not os.path.exists(src):
                    _wait(lambda: os.path.exists(src), "the external drive")
                _wait(_immich_up, "Immich")
                time.sleep(5)
        if not done:
            failed += 1
        sent += r["size"] / 1e9
        if n % 25 == 0:
            db.commit()
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {sent:.1f} GB  {sent * 1000 / el:.0f} MB/s  ({failed} failed so far)", flush=True)
    db.commit()
    print(f"upload finished: {failed} photos could not be uploaded", flush=True)


# ------------------------------------------------------------------------------------------------ albums
def album_plan(db):
    """{album name: [photo ids]}: RAW / edited, every country, every subject, and places with enough photos."""
    q = lambda sql, *a: [r[0] for r in db.execute(sql, a).fetchall()]
    base = "dup_of IS NULL"
    plan = {PREFIX + "RAW originals": q(f"SELECT id FROM photos WHERE {base} AND kind='raw'"),
            PREFIX + "Edited & JPEG": q(f"SELECT id FROM photos WHERE {base} AND kind='processed'")}
    for (c,) in db.execute(f"SELECT DISTINCT country FROM photos WHERE {base} AND country IS NOT NULL").fetchall():
        plan[f"{PREFIX}{c}" if c != "Unsorted" else PREFIX + "Unsorted (to place)"] = q(f"SELECT id FROM photos WHERE {base} AND country=?", c)
    for (s,) in db.execute(f"SELECT DISTINCT subject FROM photos WHERE {base} AND subject IS NOT NULL").fetchall():
        plan[f"{PREFIX}{s}"] = q(f"SELECT id FROM photos WHERE {base} AND subject=?", s)
    for c, p, n in db.execute(f"SELECT country, place, COUNT(*) FROM photos WHERE {base} AND place!='' AND place!='(by date)' AND country IS NOT NULL GROUP BY country, place HAVING COUNT(*)>=?", (MIN_PLACE,)).fetchall():
        plan[f"{PREFIX}{c} · {p}"] = q(f"SELECT id FROM photos WHERE {base} AND country=? AND place=?", c, p)
    return plan


def albums(dry=False):
    db = _db()
    mapped = {r[0]: r[1] for r in db.execute("SELECT photo_id, asset_id FROM immich_map")}
    plan = album_plan(db)
    s, existing = immich.call("GET", "/api/albums")
    by_name = {a["albumName"]: a["id"] for a in existing} if s == 200 else {}
    print(f"{len(plan)} albums planned; {len(mapped)} photos are in Immich", flush=True)
    for name, ids in sorted(plan.items()):
        assets = [mapped[i] for i in ids if i in mapped]
        print(f"  {name:62s} {len(assets):5d} of {len(ids)}" + ("  (exists)" if name in by_name else ""), flush=True)
        if dry or not assets:
            continue
        if name not in by_name:
            s, r = immich.call("POST", "/api/albums", {"albumName": name, "assetIds": assets[:500]})
            if s not in (200, 201):
                print("    create failed", s, str(r)[:100]); continue
            by_name[name] = r["id"]
            rest = assets[500:]
        else:
            rest = assets
        for k in range(0, len(rest), 500):
            s, r = immich.call("PUT", f"/api/albums/{by_name[name]}/assets", {"ids": rest[k:k + 500]})
            if s != 200:
                print("    add failed", s, str(r)[:100])
    print("albums done" if not dry else "dry run: nothing changed in Immich", flush=True)


def status():
    db = _db()
    n = db.execute("SELECT COUNT(*) FROM photos WHERE dup_of IS NULL").fetchone()[0]
    by = {r[0]: r[1] for r in db.execute("SELECT how, COUNT(*) FROM immich_map GROUP BY how")}
    print(f"{n} unique photos in the catalog; in Immich: {by or 'none yet'}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    if mode == "match":
        match()
    elif mode == "upload":
        upload(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif mode == "albums":
        albums("--dry" in sys.argv)
    else:
        status()
