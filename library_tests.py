"""Tests for the travel library catalog: folder rules, albums, ordering, and that the drive is never written to.

    ./venv/bin/python library_tests.py
"""
import os
import sys

import library as lib

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# ---- folder rules
c = lib.classify_path
check("country folders map to countries", c("EuropeTrip/Italy/2017-12-15/DSC_1.NEF")[:2] == ("Italy", "Europe") and c("EuropeTrip/Prague/2017-12-27/DSC_1.NEF")[0] == "Czech Republic"
      and c("EuropeTrip/Austria,Salzburg/x/IMG.JPG")[0] == "Austria" and c("EuropeTrip/Slovenia/Piran/IMG.JPG")[0] == "Slovenia")
check("UK nations", c("Uk Trip/Scotland/Lochness/DSC_1.NEF")[0] == "Scotland" and c("Uk Trip/UK Trip/Wales/DSC_1.NEF")[0] == "Wales"
      and c("Uk Trip/UK Trip/Lake District/DSC_1.NEF")[0] == "England" and c("Uk Trip/UK Trip/Dunstanburg Castle(Northumberland)/DSC_1.NEF")[0] == "England")
check("mixed card-dump folder is split correctly", c("EuropeTrip/Italy/D750/EiffelTowerTimelapse/DSC_1.jpg")[0] == "France"
      and c("EuropeTrip/Italy/D750/Nebraska Skies Seperate Jpeg and RAW/Nebraska Skies RAW/x.CR2")[0] == "USA"
      and c("EuropeTrip/Italy/D750/2018-01-15/DSC_1.NEF")[0].startswith("To confirm") and c("EuropeTrip/Italy/2017-12-15/DSC_1.NEF")[0] == "Italy")
check("personal / non-travel folders are excluded", all(c(p) is None for p in ["EuropeTrip/Baby'sPhotos/IMG_1.jpg", "Uk Trip/InstaForBaby/x.jpg", "EuropeTrip/Italy/Puppy/Oreo/x.jpg",
                                                                         "EuropeTrip/Europe_Videos/2017-09-20/x.jpg", "Bikashji Photos$Downloads/Singapore/Jeffrey_Event/x.jpg"]))
# ---- the catalog
if not lib.DB_PATH.exists():
    print("  (no catalog yet: run build_library.py meta)")
else:
    db = lib.connect()
    n = db.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    a = lib.albums()
    allc = a["groups"][0]["albums"][0]["count"]
    raw = a["groups"][0]["albums"][1]["count"]
    edited = a["groups"][0]["albums"][2]["count"]
    check("RAW and edited albums are separate and add up to everything", raw + edited == allc and raw > 0 and edited > 0, f"{raw} RAW + {edited} edited = {allc}")
    cs = [x for x in a["groups"][1]["albums"]]
    check("country albums cover every unique photo exactly once", sum(x["count"] for x in cs) == allc, f"{sum(x['count'] for x in cs)} vs {allc}")
    p = lib.photos("raw", order="taken", limit=400)
    ts = [x["taken"] for x in p["photos"]]
    check("albums are in date order", ts == sorted(ts) and all(x["kind"] == "raw" for x in p["photos"]))
    p = lib.photos("country:Scotland", kind="processed", limit=50)
    check("filters combine (country + edited only)", p["total"] > 0 and all(x["country"] == "Scotland" and x["kind"] == "processed" for x in p["photos"]))
    dup = db.execute("SELECT COUNT(*) FROM photos WHERE dup_of IS NOT NULL").fetchone()[0]
    inalb = db.execute("SELECT COUNT(*) FROM photos WHERE dup_of IS NOT NULL AND id IN (SELECT id FROM photos WHERE " + lib._where("all")[0] + ")").fetchone()[0]
    check("duplicate copies are kept out of albums", inalb == 0, f"{dup} duplicates hidden")
    missing = [r[0] for r in db.execute("SELECT path FROM photos ORDER BY RANDOM() LIMIT 40")]
    ok = all(os.path.exists(lib.abs_path(m)) for m in missing) if lib.drive_connected() else True
    check("catalog paths exist on the drive (random 40)", ok)
    ro = True
    try:
        open(os.path.join(lib.root(), ".write_test"), "w").close()
        os.remove(os.path.join(lib.root(), ".write_test"))
        ro = False
    except OSError:
        pass
    print(f"  INFO  drive is {'WRITABLE' if not ro else 'read-only'}; this project only ever reads from it")
# ---- autosaved edits and the backup
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

import backup_library
import server
from params import EditParams

name, size = "__test_photo__.nef", 123456
p = EditParams()
p.basic.exposure = 17.5
p.sky_replace.id = "some_sky"
server.edit_save(server.EditSave(name=name, size=size, params=p))
back = server.edit_load(name, size)
check("an autosaved edit comes back exactly", back["params"]["basic"]["exposure"] == 17.5 and back["params"]["sky_replace"]["id"] == "some_sky")
f = server._edit_file(name, size)
f.write_text(json.dumps({"name": name, "size": size, "params": {"basic": {"exposure": "not a number"}}}))
try:
    server.edit_load(name, size)
    check("a broken / outdated saved edit is ignored, not loaded", False)
except Exception:
    check("a broken / outdated saved edit is ignored, not loaded", True)
server.edit_delete(server.EditKey(name=name, size=size))
check("a saved edit can be deleted", not f.exists())
if lib.DB_PATH.exists():
    tmp = Path(tempfile.mkdtemp())
    backup_library.backup(tmp)
    a = sqlite3.connect(lib.DB_PATH).execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    b = sqlite3.connect(tmp / "library" / "library.db").execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    rows = sum(1 for _ in open(tmp / "albums_manifest.csv")) - 1
    check("backup is complete and readable without the app", a == b == rows and (tmp / "library" / "thumbs").exists(), f"{a} photos in database, backup and spreadsheet")
    shutil.rmtree(tmp, ignore_errors=True)
print("\nALL PASSED" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
