"""Copy the travel photos to a second place (the NAS) and prove the copy is identical. Read-only on the source drive.

    ./venv/bin/python mirror_library.py all [destination]     # copy, then verify every file by reading it back
    ./venv/bin/python mirror_library.py copy|verify|status [destination]

Every file in the catalog is copied with its folder structure (EuropeTrip/..., Uk Trip/...). While copying, a SHA-256 of the source bytes
is computed; `verify` re-reads the NAS copy and compares. Safe to stop and restart: finished files are skipped. Nothing is ever deleted.
"""
import hashlib
import os
import shutil
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import library as lib

DEFAULT_DEST = "/Volumes/NetworkShare/TravelPhotos"
CHUNK = 8 * 1024 * 1024


def _db():
    db = lib.connect()
    db.execute("CREATE TABLE IF NOT EXISTS mirror (photo_id INTEGER PRIMARY KEY, path TEXT, size INTEGER, sha256 TEXT, copied INTEGER DEFAULT 0, verified INTEGER DEFAULT 0, error TEXT)")
    db.commit()
    return db


def copy(dest: Path):
    db = _db()
    db.execute("INSERT OR REPLACE INTO meta VALUES('mirror_root', ?)", (str(dest),))
    db.commit()
    rows = db.execute("SELECT p.id, p.path, p.size FROM photos p LEFT JOIN mirror m ON m.photo_id=p.id WHERE COALESCE(m.copied,0)=0 ORDER BY p.path").fetchall()
    need = sum(r["size"] for r in rows)
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    print(f"{len(rows)} files, {need / 1e9:.1f} GB to copy; destination has {free / 1e9:.0f} GB free", flush=True)
    if free < need + 100e9:
        sys.exit("Not enough free space at the destination (keeping 100 GB spare). Nothing copied.")
    t0, done, gb = time.time(), 0, 0.0
    for r in rows:
        src, dst = lib.abs_path(r["path"]), dest / r["path"]
        err = None
        for attempt in range(3):
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                tmp = dst.with_name(dst.name + ".part")
                with open(src, "rb") as fi, open(tmp, "wb") as fo:
                    while True:
                        b = fi.read(CHUNK)
                        if not b:
                            break
                        h.update(b)
                        fo.write(b)
                    fo.flush()
                    os.fsync(fo.fileno())
                if tmp.stat().st_size != r["size"]:
                    raise IOError(f"size mismatch {tmp.stat().st_size} != {r['size']}")
                st = os.stat(src)
                os.utime(tmp, (st.st_atime, st.st_mtime))
                os.replace(tmp, dst)
                db.execute("INSERT OR REPLACE INTO mirror(photo_id,path,size,sha256,copied,verified,error) VALUES(?,?,?,?,1,0,NULL)", (r["id"], r["path"], r["size"], h.hexdigest()))
                err = None
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                time.sleep(2)
        if err:
            db.execute("INSERT OR REPLACE INTO mirror(photo_id,path,size,copied,error) VALUES(?,?,?,0,?)", (r["id"], r["path"], r["size"], err))
            print("  FAILED", r["path"], err[:100], flush=True)
        done += 1
        gb += r["size"] / 1e9
        if done % 50 == 0:
            db.commit()
            el = time.time() - t0
            print(f"  {done}/{len(rows)}  {gb:.1f} GB  {gb * 1000 / el:.0f} MB/s  ~{(need / 1e9 - gb) / max(gb / el, 1e-6) / 60:.0f} min left", flush=True)
    db.commit()
    print("copy done", flush=True)


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify(dest: Path):
    db = _db()
    rows = db.execute("SELECT photo_id, path, size, sha256 FROM mirror WHERE copied=1 AND verified=0 ORDER BY path").fetchall()
    print(f"verifying {len(rows)} files by reading them back from {dest}", flush=True)
    bad, t0, n = 0, time.time(), 0

    def work(r):
        p = dest / r["path"]
        try:
            return r, (p.stat().st_size == r["size"] and _hash_file(p) == r["sha256"])
        except OSError:
            return r, False
    with ThreadPoolExecutor(3) as pool:
        for r, ok in pool.map(work, rows):
            db.execute("UPDATE mirror SET verified=?, error=? WHERE photo_id=?", (1 if ok else 0, None if ok else "verify failed", r["photo_id"]))
            n += 1
            if not ok:
                bad += 1
                print("  MISMATCH", r["path"], flush=True)
            if n % 200 == 0:
                db.commit()
                print(f"  verified {n}/{len(rows)}  {(time.time() - t0) / 60:.0f} min", flush=True)
    db.commit()
    print(f"verify done: {len(rows) - bad} identical, {bad} problems", flush=True)


def status():
    db = _db()
    r = db.execute("SELECT COUNT(*) n, SUM(copied) c, SUM(verified) v, SUM(error IS NOT NULL AND copied=0) e FROM mirror").fetchone()
    total = db.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    print(f"{total} photos in catalog; copied {r['c'] or 0}, verified identical {r['v'] or 0}, failed {r['e'] or 0}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEFAULT_DEST)
    if mode in ("copy", "all"):
        copy(dest)
    if mode in ("verify", "all"):
        verify(dest)
    if mode == "status":
        status()
