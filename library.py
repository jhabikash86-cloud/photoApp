"""Photo library catalog: a read-only index of travel photos on the external drive, organised into virtual albums.

Nothing is copied or moved. The catalog (library/library.db, SQLite) records, for each photo on the drive: where it is, when it was
taken, with what camera, which country and place, whether it is a RAW file, an edited export or a plain JPEG, and what it shows
(subject, tags, sharpness). Albums are just queries over that table, so they cost no disk space and can never damage the originals.

build_library.py fills the catalog (resumable). The app reads it through the functions below.
"""
import glob
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LIB_DIR = Path(__file__).parent / "library"
DB_PATH = LIB_DIR / "library.db"
THUMB_DIR = LIB_DIR / "thumbs"
DEFAULT_ROOT = "/Volumes/Seagate Expansion Drive/HP_Laptop"

RAW_EXT = {".nef", ".dng", ".cr2", ".cr3", ".arw", ".nrw"}
PROC_EXT = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".heic"}

SUBJECTS = ["Landscape", "Beach & Coast", "Lakes, Rivers & Waterfalls", "Architecture", "Street & City", "Portrait", "People", "Wildlife & Animals", "Food & Details", "Night", "Indoor", "Other"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS photos (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,          -- relative to the library root
  folder TEXT, stem TEXT, ext TEXT, size INTEGER,
  kind TEXT,                          -- raw | processed
  origin TEXT,                        -- raw | edited | jpeg
  country TEXT, group_name TEXT, place TEXT,
  taken TEXT, day TEXT,
  camera TEXT, iso INTEGER, focal REAL, aperture REAL, shutter REAL, software TEXT,
  width INTEGER, height INTEGER,
  subject TEXT, tags TEXT, shares TEXT, faces INTEGER, face_frac REAL, luma REAL, sharp REAL, clip REAL,
  dup_of INTEGER,                     -- another photo that is the same shot (kept once in albums)
  raw_id INTEGER,                     -- for an edited export: the RAW it came from
  analysed INTEGER DEFAULT 0,
  thumb INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS i_country ON photos(country);
CREATE INDEX IF NOT EXISTS i_subject ON photos(subject);
CREATE INDEX IF NOT EXISTS i_taken ON photos(taken);
CREATE INDEX IF NOT EXISTS i_kind ON photos(kind);
CREATE INDEX IF NOT EXISTS i_place ON photos(place);
"""


def connect() -> sqlite3.Connection:
    LIB_DIR.mkdir(exist_ok=True)
    THUMB_DIR.mkdir(exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=60)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def root() -> str:
    """The photo library folder on the external drive. If macOS remounted the drive under another name
    ("Seagate Expansion Drive 1" after a USB hiccup), the same folder is found there automatically."""
    db = connect()
    r = db.execute("SELECT v FROM meta WHERE k='root'").fetchone()
    saved = r["v"] if r else DEFAULT_ROOT
    if os.path.isdir(saved):
        return saved
    for cand in sorted(glob.glob(f"/Volumes/*/{os.path.basename(saved)}")):
        if os.path.isdir(os.path.join(cand, "EuropeTrip")):
            return cand
    return saved


def mirror_root() -> Optional[str]:
    """Where the second copy of the photos lives (the NAS), once mirror_library.py has been run."""
    r = connect().execute("SELECT v FROM meta WHERE k='mirror_root'").fetchone()
    return r["v"] if r else None


def resolve(rel: str, roots: List[str]) -> str:
    """The first existing copy of a photo among `roots` (primary drive first); the primary path if none is reachable."""
    for r in roots:
        p = os.path.join(r, rel)
        if os.path.exists(p):
            return p
    return os.path.join(roots[0], rel)


def _roots() -> List[str]:
    m = mirror_root()
    return [root()] + ([m] if m else [])


def abs_path(rel: str) -> str:
    return resolve(rel, _roots())


def drive_connected() -> bool:
    return any(os.path.isdir(r) for r in _roots())


# ------------------------------------------------------------------------------------------------ folder rules
COUNTRY_FOLDERS = {      # first folder under EuropeTrip / Uk Trip -> (country, group)
    "italy": ("Italy", "Europe"), "prague": ("Czech Republic", "Europe"), "slovenia": ("Slovenia", "Europe"),
    "hungary": ("Hungary", "Europe"), "austria,salzburg": ("Austria", "Europe"), "scotland": ("Scotland", "United Kingdom"),
}
ENGLAND = ("dunstanburg", "england", "roseberry", "lake district", "north yorkshire", "broadchurch", "maryport", "bamburg")
NAME_HINTS = {"florence": "Italy", "praino": "Italy", "sevile": "Spain", "seville": "Spain", "prague": "Czech Republic", "venice": "Italy"}
EXCLUDE = ("baby", "instaforbaby", "europe_videos", "puppy", "oreo", "logo")


def classify_path(rel: str) -> Optional[Tuple[Optional[str], Optional[str], str]]:
    """(country, group, place) from the folder path, or None for photos that are not travel. country None = decide by date later."""
    parts = rel.split("/")
    top = parts[0]
    low = [p.lower() for p in parts]
    if any(any(x in p for x in EXCLUDE) for p in low[:-1]):
        return None
    if top == "EuropeTrip":
        sub = parts[1] if len(parts) > 2 else ""
        joined = "/".join(low)
        if "nebraska" in joined:
            return "USA", "Americas", "Nebraska skies"
        if "eiffel" in joined:
            return "France", "Europe", "Paris · Eiffel Tower timelapse"
        # card dumps from January 2018 onwards sit in 'Italy/D750', but the drive has no GPS and one of them is Paris:
        # do not guess a country for them, keep them together until the trip dates are confirmed
        if sub.lower() == "italy" and len(parts) > 4 and parts[2].lower() == "d750":
            m = re.match(r"(\d{4}-\d{2}-\d{2})", parts[3])
            if m and m.group(1) >= "2018-01-06":
                return "To confirm (Jan-Feb 2018)", "Europe", parts[3]
        c = COUNTRY_FOLDERS.get(sub.lower())
        if c:
            place = parts[2] if len(parts) > 3 else ""
            if sub.lower() == "prague":
                place = "Prague"
            if sub.lower() == "austria,salzburg":
                place = "Salzburg"
            return c[0], c[1], place if not re.fullmatch(r"[\d\-]+|d750|\d+nd750", place.lower()) else ""
        return None, "Europe", ""
    if top == "Uk Trip":
        if len(parts) < 3:
            return None if parts[-1].lower().endswith((".zip", ".rar")) else (None, "United Kingdom", "")
        sub, place = parts[1], parts[2] if len(parts) > 3 else ""
        if sub.lower() in ("instapost", "instaforbaby"):
            return None
        if sub.lower() == "scotland":
            return "Scotland", "United Kingdom", place
        if sub.lower() == "uk trip":
            if place.lower() == "wales":
                return "Wales", "United Kingdom", parts[3] if len(parts) > 4 else "Wales"
            if any(k in place.lower() for k in ENGLAND):
                return "England", "United Kingdom", place
            return "England", "United Kingdom", place
        return None, "United Kingdom", ""
    return None


# ------------------------------------------------------------------------------------------------ albums (queries)
def _where(album: str) -> Tuple[str, list]:
    """SQL condition for an album id such as 'all', 'raw', 'edited', 'country:Italy', 'subject:Landscape', 'place:Italy|Florence', 'day:2017-12-15'."""
    base = "dup_of IS NULL"
    kind, _, val = album.partition(":")
    if kind == "all":
        return base, []
    if kind == "raw":
        return f"{base} AND kind='raw'", []
    if kind == "edited":
        return f"{base} AND kind='processed'", []
    if kind == "country":
        c = val.split("|")
        return f"{base} AND country=?" + (" AND kind=?" if len(c) > 1 else ""), c
    if kind == "subject":
        return f"{base} AND subject=?", [val]
    if kind == "place":
        c, p = val.split("|", 1)
        return f"{base} AND country=? AND place=?", [c, p]
    if kind == "day":
        c, d = val.split("|", 1)
        return f"{base} AND country=? AND day=?", [c, d]
    if kind == "tag":
        return f"{base} AND tags LIKE ?", [f'%"{val}"%']
    raise ValueError("unknown album")


def albums() -> dict:
    """The album tree with counts: {groups: [{title, albums: [{id, title, count, raw, edited}]}]}"""
    db = connect()
    q = lambda sql, *a: db.execute(sql, a).fetchall()
    base = "dup_of IS NULL"

    def entry(id_, title, where, args=()):
        r = db.execute(f"SELECT COUNT(*) n, SUM(kind='raw') r, SUM(kind='processed') e FROM photos WHERE {where}", args).fetchone()
        return {"id": id_, "title": title, "count": r["n"] or 0, "raw": r["r"] or 0, "edited": r["e"] or 0}
    out = []
    out.append({"title": "Collections", "albums": [entry("all", "All travel photos", base), entry("raw", "RAW (Nikon D750 originals)", f"{base} AND kind='raw'"),
                                                    entry("edited", "Edited & JPEG", f"{base} AND kind='processed'")]})
    countries = q(f"SELECT country, group_name, COUNT(*) n FROM photos WHERE {base} AND country IS NOT NULL GROUP BY country ORDER BY group_name, n DESC")
    out.append({"title": "By country", "albums": [dict(entry(f"country:{r['country']}", f"{r['country']}", f"{base} AND country=?", (r["country"],)), group=r["group_name"]) for r in countries]})
    subj = {r["subject"]: r["n"] for r in q(f"SELECT subject, COUNT(*) n FROM photos WHERE {base} AND subject IS NOT NULL GROUP BY subject")}
    out.append({"title": "By subject", "albums": [entry(f"subject:{s}", s, f"{base} AND subject=?", (s,)) for s in SUBJECTS if subj.get(s)]})
    places = q(f"SELECT country, place, COUNT(*) n FROM photos WHERE {base} AND place!='' AND country IS NOT NULL GROUP BY country, place HAVING n>=8 ORDER BY country, MIN(taken)")
    out.append({"title": "By place", "albums": [entry(f"place:{r['country']}|{r['place']}", f"{r['country']} · {r['place']}", f"{base} AND country=? AND place=?", (r["country"], r["place"])) for r in places]})
    try:
        in_immich = db.execute("SELECT COUNT(*) FROM immich_map").fetchone()[0]
    except Exception:
        in_immich = 0
    return {"groups": out, "root": root(), "connected": drive_connected(), "immich_copies": in_immich}


def photos(album: str, order: str = "taken", limit: int = 120, offset: int = 0, kind: Optional[str] = None, subject: Optional[str] = None) -> dict:
    where, args = _where(album)
    if kind in ("raw", "processed"):
        where += " AND kind=?"
        args = list(args) + [kind]
    if subject:
        where += " AND subject=?"
        args = list(args) + [subject]
    db = connect()
    total = db.execute(f"SELECT COUNT(*) FROM photos WHERE {where}", args).fetchone()[0]
    od = {"taken": "taken, path", "newest": "taken DESC, path", "sharp": "sharp DESC", "name": "path"}.get(order, "taken, path")
    rows = db.execute(f"SELECT id, path, kind, origin, country, place, taken, subject, tags, camera, sharp, width, height, thumb FROM photos WHERE {where} ORDER BY {od} LIMIT ? OFFSET ?",
                      list(args) + [limit, offset]).fetchall()
    return {"total": total, "photos": [dict(r) for r in rows]}


def photo(pid: int) -> Optional[dict]:
    r = connect().execute("SELECT * FROM photos WHERE id=?", (pid,)).fetchone()
    return dict(r) if r else None
