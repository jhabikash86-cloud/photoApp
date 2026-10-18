"""Back up (or restore) everything that makes up your albums and edits, to any folder: another drive, the NAS, a USB stick.

    ./venv/bin/python backup_library.py backup [folder]      # default folder: ~/NeuralEditPro-backup
    ./venv/bin/python backup_library.py restore [folder]     # put a backup back (asks nothing, never touches the photos themselves)

What is saved: the catalog (library.db), the place rules, the thumbnails, your autosaved edits (edits/), and albums_manifest.csv: a plain
spreadsheet of every photo's country, place, subject, date and RAW/edited status, readable without this app.
The photos on the external drive are only ever read. Safe to run repeatedly (only new thumbnails are copied).
"""
import csv
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
LIB = HERE / "library"
EDITS = HERE / "edits"
DEFAULT = Path.home() / "NeuralEditPro-backup"


def backup(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "library").mkdir(exist_ok=True)
    # consistent copy of the database even if the app is running
    src = sqlite3.connect(LIB / "library.db")
    out = sqlite3.connect(dest / "library" / "library.db")
    src.backup(out)
    out.close()
    n = 0
    for f in ("place_rules.json",):
        if (LIB / f).exists():
            shutil.copy2(LIB / f, dest / "library" / f)
    (dest / "library" / "thumbs").mkdir(exist_ok=True)
    for t in (LIB / "thumbs").glob("*.jpg"):
        d = dest / "library" / "thumbs" / t.name
        if not d.exists() or d.stat().st_size != t.stat().st_size:
            shutil.copy2(t, d)
            n += 1
    ne = 0
    if EDITS.exists():
        (dest / "edits").mkdir(exist_ok=True)
        for e in EDITS.glob("*.json"):
            shutil.copy2(e, dest / "edits" / e.name)
            ne += 1
    db = sqlite3.connect(LIB / "library.db")
    db.row_factory = sqlite3.Row
    with open(dest / "albums_manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path_on_drive", "country", "place", "subject", "taken", "kind", "origin", "camera", "duplicate_of_row", "location_confidence"])
        for r in db.execute("SELECT path, country, place, subject, taken, kind, origin, camera, dup_of, loc_conf FROM photos ORDER BY country, taken"):
            w.writerow(list(r))
    (dest / "BACKUP_INFO.txt").write_text(f"Neural Edit Pro backup, {time.strftime('%Y-%m-%d %H:%M')}\nPhotos live on the external drive (HP_Laptop); this folder holds the albums catalog, thumbnails and edits.\n"
                                           "Restore with: ./venv/bin/python backup_library.py restore <this folder>\n")
    total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"backup to {dest}: database, {n} new thumbnails, {ne} edits, manifest. Folder size {total / 1e6:.0f} MB")


def restore(src: Path):
    if not (src / "library" / "library.db").exists():
        sys.exit(f"{src} does not look like a backup (no library/library.db).")
    LIB.mkdir(exist_ok=True)
    shutil.copy2(src / "library" / "library.db", LIB / "library.db")
    if (src / "library" / "place_rules.json").exists():
        shutil.copy2(src / "library" / "place_rules.json", LIB / "place_rules.json")
    (LIB / "thumbs").mkdir(exist_ok=True)
    for t in (src / "library" / "thumbs").glob("*.jpg"):
        if not (LIB / "thumbs" / t.name).exists():
            shutil.copy2(t, LIB / "thumbs" / t.name)
    if (src / "edits").exists():
        EDITS.mkdir(exist_ok=True)
        for e in (src / "edits").glob("*.json"):
            shutil.copy2(e, EDITS / e.name)
    print("restored the catalog, thumbnails and edits.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "backup"
    folder = Path(os.path.expanduser(sys.argv[2])) if len(sys.argv) > 2 else DEFAULT
    {"backup": backup, "restore": restore}.get(mode, lambda _: sys.exit(__doc__))(folder)
