"""Build / update the travel photo catalog (library/library.db) from the external drive. READ-ONLY on the drive.

    ./venv/bin/python build_library.py meta         # stage A: find travel photos, read dates / camera / country (minutes)
    ./venv/bin/python build_library.py analyse      # stage B: what each photo shows + thumbnails (about 0.6 s per photo, resumable)
    ./venv/bin/python build_library.py report       # print what is in the catalog

Stage B can be stopped and restarted at any time; finished photos are skipped.
"""
import io
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import library as lib

ROOT = lib.DEFAULT_ROOT


# ------------------------------------------------------------------------------------------------ stage A
def _exif(path: str) -> dict:
    """Date, camera, exposure from the file's EXIF (PIL reads NEF / DNG / CR2 / JPEG headers cheaply)."""
    from PIL import Image
    out = {}
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".nef", ".dng", ".cr2", ".cr3", ".arw", ".nrw", ".jpg", ".jpeg") or os.path.getsize(path) > 120_000_000:
        return out          # big TIFF / PNG exports: opening them reads hundreds of MB; they take date and camera from their RAW instead
    try:
        im = Image.open(path)
        ex = im.getexif()
        sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
        d = sub.get(36867) or ex.get(306)
        if d:
            out["taken"] = datetime.strptime(str(d).strip()[:19], "%Y:%m:%d %H:%M:%S").isoformat(sep=" ")
        make, model = str(ex.get(271) or "").strip(), str(ex.get(272) or "").strip()
        out["camera"] = (model if model.startswith(make.split(" ")[0]) else f"{make} {model}").strip().title().replace("Nikon Corporation ", "")
        out["iso"] = sub.get(34855)
        for k, tag in (("focal", 37386), ("aperture", 33437), ("shutter", 33434)):
            v = sub.get(tag)
            out[k] = float(v) if v is not None else None
        out["software"] = str(ex.get(305) or "")
        if im.size[0] > 400:
            out["width"], out["height"] = im.size
    except Exception:
        pass
    return out


def stage_meta(limit=None):
    cat = os.path.join(os.path.dirname(os.path.abspath(__file__)), "library", "catalog.jsonl")
    # the file list comes from a quick directory walk of the travel folders (no file is opened except for EXIF)
    files = []
    for top in ("EuropeTrip", "Uk Trip"):
        for dp, dn, fn in os.walk(os.path.join(ROOT, top)):
            dn[:] = [d for d in dn if not d.startswith(".")]
            for f in fn:
                ext = os.path.splitext(f)[1].lower()
                if f.startswith(".") or (ext not in lib.RAW_EXT and ext not in lib.PROC_EXT):
                    continue
                rel = os.path.relpath(os.path.join(dp, f), ROOT).replace(os.sep, "/")
                info = lib.classify_path(rel)
                if info is None:
                    continue
                files.append((rel, ext, info))
    if limit:
        files = files[:limit]
    print(f"{len(files)} travel images found", flush=True)
    db = lib.connect()
    db.execute("INSERT OR REPLACE INTO meta VALUES('root', ?)", (ROOT,))
    have = {r[0] for r in db.execute("SELECT path FROM photos")}
    todo = [f for f in files if f[0] not in have]
    print(f"{len(todo)} new to read", flush=True)
    t0 = time.time()

    def work(item):
        rel, ext, info = item
        p = os.path.join(ROOT, rel)
        st = os.stat(p)
        e = _exif(p)
        return rel, ext, info, st, e
    done = 0
    with ThreadPoolExecutor(3) as pool:
        for rel, ext, (country, group, place), st, e in pool.map(work, todo):
            taken = e.get("taken") or datetime.fromtimestamp(st.st_mtime).isoformat(sep=" ")
            stem = os.path.splitext(os.path.basename(rel))[0]
            kind = "raw" if ext in lib.RAW_EXT else "processed"
            db.execute("INSERT OR IGNORE INTO photos(path,folder,stem,ext,size,kind,country,group_name,place,taken,day,camera,iso,focal,aperture,shutter,software,width,height) "
                       "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (rel, os.path.dirname(rel), stem, ext, st.st_size, kind, country, group, place, taken, taken[:10], e.get("camera"), e.get("iso"), e.get("focal"),
                        e.get("aperture"), e.get("shutter"), e.get("software"), e.get("width"), e.get("height")))
            done += 1
            if done % 500 == 0:
                db.commit()
                print(f"  {done}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    db.commit()
    organise(db)
    print("meta done", flush=True)


EDIT_WORDS = ("edit", "psd", "-hdr", "-pano", "_hdr", "_pano", "lightroom", "final")
EDIT_SOFTWARE = ("adobe", "lightroom", "photoshop", "capture nx", "gimp", "snapseed", "photos", "nik", "luminar", "affinity")


def reapply_paths(db):
    """Re-run the folder rules over everything already in the catalog (rules improve as we learn the folder layout)."""
    gone = 0
    for r in db.execute("SELECT id, path FROM photos").fetchall():
        info = lib.classify_path(r["path"])
        if info is None:
            db.execute("DELETE FROM photos WHERE id=?", (r["id"],))
            gone += 1
            continue
        country, group, place = info
        db.execute("UPDATE photos SET country=?, group_name=?, place=? WHERE id=?", (country, group, place, r["id"]))
    db.commit()
    print(f"re-applied folder rules; {gone} non-travel photos dropped", flush=True)


def apply_place_rules(db):
    """Country / place for photos whose folder does not say where they were taken, from library/place_rules.json: time windows that were
    identified by looking at the photos (landmarks, signs). Each rule carries a confidence so doubtful ones can be reviewed."""
    f = lib.LIB_DIR / "place_rules.json"
    cols = {r[1] for r in db.execute("PRAGMA table_info(photos)")}
    if "loc_conf" not in cols:
        db.execute("ALTER TABLE photos ADD COLUMN loc_conf TEXT")
    db.execute("UPDATE photos SET loc_conf=NULL")
    if not f.exists():
        return
    n = 0
    for rule in json.loads(f.read_text()):
        cur = db.execute("UPDATE photos SET country=?, group_name=?, place=?, loc_conf=? WHERE (country LIKE 'To confirm%' OR country='Unsorted') AND taken BETWEEN ? AND ?",
                         (rule["country"], rule.get("group", "Europe"), rule["place"], rule.get("conf", "medium"), rule["from"] + ":00", rule["to"] + ":59"))
        n += cur.rowcount
    db.commit()
    # an edited export that matches a RAW takes the RAW's country and place
    db.execute("""UPDATE photos SET country=(SELECT country FROM photos r WHERE r.id=photos.raw_id), group_name=(SELECT group_name FROM photos r WHERE r.id=photos.raw_id),
                  place=(SELECT place FROM photos r WHERE r.id=photos.raw_id), loc_conf=(SELECT loc_conf FROM photos r WHERE r.id=photos.raw_id)
                  WHERE raw_id IS NOT NULL AND country='Unsorted' AND (SELECT country FROM photos r WHERE r.id=photos.raw_id)!='Unsorted'""")
    db.commit()
    print(f"place rules placed {n} photos", flush=True)


def organise(db):
    reapply_paths(db)
    """Everything derived from stage A: edited vs plain JPEG, RAW pairs, duplicates, countries of loose photos."""
    rows = db.execute("SELECT id, path, folder, stem, kind, software, taken, size, country, ext FROM photos").fetchall()
    raw_by_stem = defaultdict(list)
    for r in rows:
        if r["kind"] == "raw":
            raw_by_stem[r["stem"].lower()].append(r)

    def strip_edit(stem: str) -> str:
        return re.sub(r"(-hdr|_hdr|-pano(-\d+)?|_pano|-edit(ed)?|_edit(ed)?|-\d+|\s*\(\d+\)|-copy|_copy)$", "", stem.lower())
    # --- origin + RAW pairing
    for r in rows:
        if r["kind"] == "raw":
            db.execute("UPDATE photos SET origin='raw', raw_id=NULL WHERE id=?", (r["id"],))
            continue
        low = r["path"].lower()
        stem = r["stem"].lower()
        partner = raw_by_stem.get(stem) or raw_by_stem.get(strip_edit(stem))
        in_edit_folder = any(w in low for w in ("edit", "psd_edited")) and "/d750/" not in low
        by_software = any(s in (r["software"] or "").lower() for s in EDIT_SOFTWARE)
        by_name = any(w in stem for w in EDIT_WORDS)
        origin = "edited" if (in_edit_folder or by_software or by_name or (partner and by_software)) else "jpeg"
        raw_id = None
        if partner:
            same_day = [p for p in partner if p["taken"][:10] == r["taken"][:10]]
            raw_id = (same_day or partner)[0]["id"]
            if origin == "jpeg" and (r["software"] or in_edit_folder):
                origin = "edited"
        db.execute("UPDATE photos SET origin=?, raw_id=? WHERE id=?", (origin, raw_id, r["id"]))
        if raw_id:
            db.execute("""UPDATE photos SET camera=COALESCE(camera,(SELECT camera FROM photos x WHERE x.id=?)), iso=COALESCE(iso,(SELECT iso FROM photos x WHERE x.id=?)),
                          focal=COALESCE(focal,(SELECT focal FROM photos x WHERE x.id=?)), taken=(SELECT taken FROM photos x WHERE x.id=?), day=substr((SELECT taken FROM photos x WHERE x.id=?),1,10)
                          WHERE id=?""", (raw_id,) * 5 + (r["id"],))
    # --- duplicates: the same shot copied into several folders (same name, size and capture time): keep the first
    seen = {}
    dups = 0
    for r in db.execute("SELECT id, stem, size, taken, ext, path FROM photos ORDER BY (folder LIKE '%/D750%'), path").fetchall():
        key = (r["stem"].lower(), r["size"], r["taken"], r["ext"])
        if key in seen:
            db.execute("UPDATE photos SET dup_of=? WHERE id=?", (seen[key], r["id"]))
            dups += 1
        else:
            seen[key] = r["id"]
            db.execute("UPDATE photos SET dup_of=NULL WHERE id=?", (r["id"],))
    # --- country for loose photos (no country folder): by capture date against the dated country folders, else by file name
    ranges = {}
    for c, in db.execute("SELECT DISTINCT country FROM photos WHERE country IS NOT NULL").fetchall():
        days = sorted(d for d, in db.execute("SELECT day FROM photos WHERE country=? AND taken NOT LIKE '1970%' AND dup_of IS NULL", (c,)).fetchall())
        if len(days) > 20:
            ranges[c] = (days[int(len(days) * 0.02)], days[int(len(days) * 0.98)])
    loose = db.execute("SELECT id, stem, day, group_name FROM photos WHERE country IS NULL").fetchall()
    for r in loose:
        c = None
        for k, v in lib.NAME_HINTS.items():
            if k in r["stem"].lower():
                c = v
        if c is None:
            hits = [k for k, (a, b) in ranges.items() if a <= r["day"] <= b]
            if len(hits) == 1:
                c = hits[0]
        db.execute("UPDATE photos SET country=? , place=CASE WHEN place='' THEN '(by date)' ELSE place END WHERE id=?", (c or "Unsorted", r["id"]))
    db.commit()
    apply_place_rules(db)
    print(f"organised: {dups} duplicates, {len(loose)} loose photos placed by date/name", flush=True)


# ------------------------------------------------------------------------------------------------ stage B
EXTRA = {
    "animal": ["animal"], "vehicles": ["car", "bus", "truck", "van", "minibike", "bicycle", "airplane"], "boats": ["boat", "ship"],
    "street_furniture": ["streetlight", "traffic light", "signboard", "awning", "pole", "bench", "stairs", "stairway"],
    "road": ["road", "sidewalk", "path", "dirt track", "runway"], "earth": ["earth", "land", "sand", "dirt track"], "sand": ["sand"],
    "sea": ["sea", "water", "river", "lake", "waterfall"], "sea_only": ["sea"], "lake_only": ["lake"], "river_only": ["river"], "fall_only": ["waterfall"], "art": ["sculpture", "fountain", "statue", "monument"], "food": ["food", "plate", "bottle", "glass"],
}


def _ids():
    import semseg
    semseg._load()
    labels = {int(k): v.strip() for k, v in json.load(open(semseg.CONFIG))["id2label"].items()}
    by = {v: k for k, v in labels.items()}
    ids = {g: [by[n] for n in names if n in by] for g, names in semseg.GROUPS.items()}
    ids.update({g: [by[n] for n in names if n in by] for g, names in EXTRA.items()})
    ids["indoor_objects"] = [by[n] for n in semseg.INDOOR if n in by]
    return ids


def apply_orient(im, o):
    """EXIF orientation: 3 = upside down, 6 = rotate 90 clockwise, 8 = rotate 90 counter-clockwise."""
    return {3: lambda i: i.rotate(180), 6: lambda i: i.rotate(-90, expand=True), 8: lambda i: i.rotate(90, expand=True)}.get(o or 1, lambda i: i)(im)


def _preview(path: str, orient=None):
    """A <=1000 px RGB preview, read cheaply: the JPEG embedded in a RAW file, or a draft decode of a JPEG."""
    import numpy as np
    from PIL import Image
    ext = os.path.splitext(path)[1].lower()
    im = None
    if ext in lib.RAW_EXT:
        import rawpy
        with rawpy.imread(path) as r:
            try:
                th = r.extract_thumb()
                im = Image.open(io.BytesIO(th.data)) if th.format == rawpy.ThumbFormat.JPEG else Image.fromarray(th.data)
            except (rawpy.LibRawUnsupportedThumbnailError, rawpy.LibRawNoThumbnailError):
                # e.g. iPhone ProRAW DNG: no usable embedded preview, so develop it quickly at half size
                rgb = r.postprocess(half_size=True, use_camera_wb=True, no_auto_bright=False, output_bps=8)
                im = Image.fromarray(rgb)
    else:
        im = Image.open(path)
    size = im.size
    im.draft("RGB", (1000, 1000))
    im = im.convert("RGB")
    im.thumbnail((1000, 1000))
    return apply_orient(im, orient), size


def decide(sh: dict, face_frac: float, luma: float, clip: float) -> str:
    """The subject of a photo from its scene shares (fractions of the frame), face size and brightness."""
    g = lambda k: sh.get(k, 0.0)
    nature = g("sky") + g("trees") + g("grass") + g("water") + g("mountains") + g("earth")
    street_stuff = g("vehicles") + g("street_furniture") + g("road")
    person = g("person")
    sea = g("sea_only") if "sea_only" in sh else max(g("sea"), g("water"))
    fresh = g("lake_only") + g("river_only") + g("fall_only")
    if luma < 0.12 and clip < 0.2:
        return "Night"
    if face_frac >= 0.02:
        return "Portrait"
    if face_frac >= 0.004 or person >= 0.03:
        return "People"
    if g("animal") >= 0.02:
        return "Wildlife & Animals"
    if g("indoor_objects") > 0.14 and g("sky") < 0.04:
        return "Indoor"
    if g("food") >= 0.05:
        return "Food & Details"
    if sea >= 0.10 or (g("sand") >= 0.03 and sea + fresh >= 0.08):
        return "Beach & Coast"
    if "sea_only" in sh and fresh >= 0.10:
        return "Lakes, Rivers & Waterfalls"
    if "sea_only" not in sh and max(g("sea"), g("water")) >= 0.12:
        return "Beach & Coast"            # old-style record: water not yet split into sea / lake / river
    if g("buildings") >= 0.30:
        return "Architecture"
    if g("road") >= 0.22 or g("vehicles") >= 0.08 or (g("buildings") >= 0.12 and street_stuff >= 0.10):
        return "Street & City"
    if g("buildings") >= 0.12:
        return "Architecture"
    if nature >= 0.5:
        return "Landscape"
    return "Other"


def classify(a, ids, hsv_fn):
    """Subject, tags and numbers for one preview (float32 RGB 0..1)."""
    import cv2
    import numpy as np
    import scene
    import semseg
    probs = semseg._scale_probs(a, 512, False)
    lab = probs.argmax(axis=0)
    sh = {g: float(np.isin(lab, i).mean()) for g, i in ids.items()}
    faces = scene.detect_faces(a)
    face_frac = max([f.area_frac for f in faces], default=0.0)
    y = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    luma = float(np.median(y))
    clip = float((a.max(axis=2) > 0.99).mean())
    g = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(cv2.resize(g, (800, int(800 * g.shape[0] / g.shape[1]))), cv2.CV_64F).var())
    subject = decide(sh, face_frac, luma, clip)
    sea = sh["sea"]
    tags = []
    if sh["mountains"] >= 0.12: tags.append("Mountains")
    if sea >= 0.08: tags.append("Water")
    if sh["trees"] >= 0.25: tags.append("Trees & Forest")
    if sh["sky"] >= 0.45: tags.append("Big sky")
    if sh["buildings"] >= 0.15 and subject != "Architecture": tags.append("Buildings")
    if sh["animal"] >= 0.02 and subject != "Wildlife & Animals": tags.append("Animals")
    if sh["art"] >= 0.03: tags.append("Statues & fountains")
    if sh["vehicles"] >= 0.03: tags.append("Vehicles")
    skyp = cv2.resize((lab == ids["sky"][0]).astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    if skyp.mean() > 0.05:
        hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
        h, s = hsv[..., 0][skyp], hsv[..., 1][skyp]
        if float((((h < 50) | (h > 320)) & (s > 0.25)).mean()) > 0.3:
            tags.append("Golden / sunset sky")
    if sh["sea_only"] + sh["lake_only"] + sh["river_only"] >= 0.08 and "Water" not in tags: tags.append("Water")
    if sh["fall_only"] >= 0.02: tags.append("Waterfall")
    if sh["sand"] >= 0.03: tags.append("Sand & beach")
    return subject, tags, {k: round(v, 3) for k, v in sh.items()}, len(faces), face_frac, luma, sharp, clip


def stage_analyse(limit=None):
    import numpy as np
    from PIL import Image
    db = lib.connect()
    ids = _ids()
    todo = db.execute("SELECT id, path, kind, raw_id FROM photos WHERE analysed=0 AND dup_of IS NULL ORDER BY (raw_id IS NOT NULL), id").fetchall()
    # an edited export of a RAW that is in the catalog inherits the RAW's subject (it shows the same thing)
    todo = [r for r in todo if not r["raw_id"]]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} photos to analyse", flush=True)
    t0 = time.time()
    n = 0
    for r in todo:
        p = lib.abs_path(r["path"])
        try:
            im, size = _preview(p)
            a = np.asarray(im, np.float32) / 255
            subject, tags, shares, nf, ff, luma, sharp, clip = classify(a, ids, None)
            th = im.copy(); th.thumbnail((480, 480))
            th.save(lib.THUMB_DIR / f"{r['id']}.jpg", "JPEG", quality=82)
            db.execute("UPDATE photos SET subject=?,tags=?,shares=?,faces=?,face_frac=?,luma=?,sharp=?,clip=?,analysed=1,thumb=1,width=COALESCE(width,?),height=COALESCE(height,?) WHERE id=?",
                       (subject, json.dumps(tags), json.dumps(shares), nf, ff, luma, sharp, clip, size[0], size[1], r["id"]))
        except Exception as e:
            db.execute("UPDATE photos SET analysed=-1 WHERE id=?", (r["id"],))
            print("  skip", r["path"], type(e).__name__, str(e)[:80], flush=True)
        n += 1
        if n % 25 == 0:
            db.commit()
        if n % 200 == 0:
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)}  {rate:.2f}s each  ~{(len(todo) - n) * rate / 60:.0f} min left", flush=True)
    db.commit()
    inherit(db)
    print("analyse done", flush=True)


def inherit(db):
    """Edited exports take the subject / tags of the RAW they came from."""
    db.execute("""UPDATE photos SET subject=(SELECT subject FROM photos r WHERE r.id=photos.raw_id), tags=(SELECT tags FROM photos r WHERE r.id=photos.raw_id),
                  faces=(SELECT faces FROM photos r WHERE r.id=photos.raw_id), luma=(SELECT luma FROM photos r WHERE r.id=photos.raw_id)
                  WHERE raw_id IS NOT NULL AND subject IS NULL""")
    db.execute("UPDATE photos SET subject=(SELECT subject FROM photos o WHERE o.id=photos.dup_of), tags=(SELECT tags FROM photos o WHERE o.id=photos.dup_of) WHERE dup_of IS NOT NULL AND subject IS NULL")
    db.commit()


def _orient(path: str) -> int:
    from PIL import Image
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff", ".png") or os.path.getsize(path) > 120_000_000:
        return 1
    try:
        return int(Image.open(path).getexif().get(274) or 1)
    except Exception:
        return 1


def relabel(db):
    """Re-apply the subject rules to every analysed photo from its stored numbers (no drive access)."""
    n = 0
    for r in db.execute("SELECT id, shares, face_frac, luma, clip, subject FROM photos WHERE analysed=1 AND shares IS NOT NULL").fetchall():
        sh = json.loads(r["shares"])
        sub = decide(sh, r["face_frac"] or 0.0, r["luma"] if r["luma"] is not None else 0.5, r["clip"] or 0.0)
        if sub != r["subject"]:
            db.execute("UPDATE photos SET subject=? WHERE id=?", (sub, r["id"]))
            n += 1
    db.commit()
    db.execute("""UPDATE photos SET subject=(SELECT subject FROM photos r WHERE r.id=photos.raw_id) WHERE raw_id IS NOT NULL""")
    db.execute("""UPDATE photos SET subject=(SELECT subject FROM photos o WHERE o.id=photos.dup_of) WHERE dup_of IS NOT NULL""")
    db.commit()
    print(f"relabelled {n} photos from stored data", flush=True)


def stage_refine(limit=None):
    """Fix sideways previews and split water into sea / lake / river / waterfall. Re-reads only the photos that need it."""
    import numpy as np
    db = lib.connect()
    cols = {r[1] for r in db.execute("PRAGMA table_info(photos)")}
    if "orient" not in cols:
        db.execute("ALTER TABLE photos ADD COLUMN orient INTEGER")
    if "v2" not in cols:
        db.execute("ALTER TABLE photos ADD COLUMN v2 INTEGER DEFAULT 0")
    db.commit()
    todo = db.execute("SELECT id, path FROM photos WHERE orient IS NULL AND dup_of IS NULL").fetchall()
    print(f"reading orientation of {len(todo)} photos", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(2) as pool:
        for k, (r, o) in enumerate(zip(todo, pool.map(lambda r: _orient(lib.abs_path(r["path"])), todo)), 1):
            db.execute("UPDATE photos SET orient=? WHERE id=?", (o, r["id"]))
            if k % 1000 == 0:
                db.commit()
                print(f"  {k}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    db.commit()
    # who must be looked at again: sideways previews (faces and thumbnails were wrong) and anything water-ish or unclassified
    cand = []
    for r in db.execute("SELECT id, path, orient, shares, subject FROM photos WHERE analysed=1 AND raw_id IS NULL AND dup_of IS NULL AND v2=0").fetchall():
        sh = json.loads(r["shares"]) if r["shares"] else {}
        wet = sh.get("sea", 0) >= 0.05 or sh.get("water", 0) >= 0.06
        if (r["orient"] or 1) != 1 or wet or r["subject"] in ("Other", "Beach & Coast", "Indoor"):
            cand.append(r)
    if limit:
        cand = cand[:limit]
    print(f"{len(cand)} photos to re-analyse (sideways: {sum(1 for r in cand if (r['orient'] or 1) != 1)})", flush=True)
    ids = _ids()
    t0 = time.time()
    for n, r in enumerate(cand, 1):
        try:
            im, size = _preview(lib.abs_path(r["path"]), r["orient"])
            a = np.asarray(im, np.float32) / 255
            subject, tags, shares, nf, ff, luma, sharp, clip = classify(a, ids, None)
            th = im.copy(); th.thumbnail((480, 480))
            th.save(lib.THUMB_DIR / f"{r['id']}.jpg", "JPEG", quality=82)
            db.execute("UPDATE photos SET subject=?,tags=?,shares=?,faces=?,face_frac=?,luma=?,sharp=?,clip=?,v2=1 WHERE id=?",
                       (subject, json.dumps(tags), json.dumps(shares), nf, ff, luma, sharp, clip, r["id"]))
        except Exception as e:
            print("  skip", r["path"], type(e).__name__, str(e)[:60], flush=True)
        if n % 25 == 0:
            db.commit()
        if n % 200 == 0:
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(cand)}  {rate:.2f}s each  ~{(len(cand) - n) * rate / 60:.0f} min left", flush=True)
    db.commit()
    relabel(db)
    print("refine done", flush=True)


def report():
    db = lib.connect()
    q = lambda s, *a: db.execute(s, a).fetchall()
    tot = q("SELECT COUNT(*) n, SUM(kind='raw') r, SUM(kind='processed') p, SUM(dup_of IS NOT NULL) d, SUM(analysed=1) a, SUM(size) s FROM photos")[0]
    print(f"{tot['n']} photos: {tot['r']} RAW, {tot['p']} processed, {tot['d']} duplicates, {tot['a']} analysed, {tot['s'] / 1e9:.0f} GB")
    print("origin:", {r['origin']: r['n'] for r in q("SELECT origin, COUNT(*) n FROM photos WHERE dup_of IS NULL GROUP BY origin")})
    print("\nBy country (unique photos):")
    for r in q("SELECT country, group_name, SUM(kind='raw') r, SUM(kind='processed') p, MIN(day) a, MAX(day) b FROM photos WHERE dup_of IS NULL GROUP BY country ORDER BY group_name, r+p DESC"):
        print(f"  {str(r['country']):16s} {r['group_name'] or '':15s} RAW {r['r']:5d}  edited/jpeg {r['p']:5d}   {r['a']} .. {r['b']}")
    print("\nBy subject:")
    for r in q("SELECT subject, COUNT(*) n FROM photos WHERE dup_of IS NULL GROUP BY subject ORDER BY n DESC"):
        print(f"  {str(r['subject']):22s} {r['n']}")
    print("\nCameras:")
    for r in q("SELECT camera, COUNT(*) n FROM photos WHERE dup_of IS NULL GROUP BY camera ORDER BY n DESC LIMIT 8"):
        print(f"  {str(r['camera']):28s} {r['n']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if cmd == "meta":
        stage_meta(lim)
    elif cmd == "analyse":
        stage_analyse(lim)
    elif cmd == "refine":
        stage_refine(lim)
    elif cmd == "relabel":
        relabel(lib.connect())
    else:
        report()
