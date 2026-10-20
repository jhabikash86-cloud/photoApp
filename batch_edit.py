"""Edit a whole folder without opening the app: the hours-saver for a card dump.

    ./venv/bin/python batch_edit.py <folder> [--out DIR] [--look drama|auto] [--preset web|full|social|print|png16|tiny]
                                    [--straighten] [--skip-blurry] [--recursive] [--limit N]

For every photo (RAW, HEIC, JPEG, ...) it analyses the scene, builds the edit (Auto, or the Drama look that fits THAT photo), optionally levels a crooked
horizon, writes the result with the chosen export preset, and moves on. Photos that already have an output are skipped, so an interrupted run resumes.
A blurry shot is only flagged (or skipped with --skip-blurry); nothing in the source folder is ever changed.
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import drama_looks
import engine
import exporter
import geometry
from params import Geometry

EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".nef", ".nrw", ".dng", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", default=None, help="where to write (default: <folder>/edited)")
    ap.add_argument("--look", default="drama", choices=["drama", "auto"])
    ap.add_argument("--preset", default="web", choices=list(exporter.PRESETS))
    ap.add_argument("--straighten", action="store_true")
    ap.add_argument("--skip-blurry", action="store_true")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    src = Path(a.folder).expanduser()
    out = Path(a.out).expanduser() if a.out else src / "edited"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in (src.rglob("*") if a.recursive else src.glob("*")) if p.suffix.lower() in EXT and out not in p.parents)
    # a RAW+JPEG pair is the same shot: edit the RAW
    raw_stems = {p.stem.lower() for p in files if p.suffix.lower() in {".nef", ".nrw", ".dng", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2"}}
    files = [p for p in files if not (p.suffix.lower() in {".jpg", ".jpeg"} and p.stem.lower() in raw_stems)]
    if a.limit:
        files = files[: a.limit]
    print(f"{len(files)} photos -> {out}  (look: {a.look}, preset: {a.preset}{', straighten' if a.straighten else ''})", flush=True)
    t0, done, skipped, failed = time.time(), 0, 0, 0
    for n, f in enumerate(files, 1):
        stem = f"{f.stem}-{a.preset}"
        if any(out.glob(f"{stem}.*")):
            skipped += 1
            continue
        try:
            an = engine.analyze(str(f), f"batch{n}")
            q = an.summary().get("quality", {})
            if a.skip_blurry and q.get("blur") == "blurry":
                print(f"[{n}/{len(files)}] {f.name}: blurry, skipped", flush=True)
                skipped += 1
                continue
            base = engine.auto_params(an, "auto")
            if a.look == "drama":
                lid = drama_looks.generate(an)[0]["id"]
                params, _, _ = drama_looks.build(an, base, lid, 1.0)
            else:
                lid, params = "auto", base
            note = ""
            if a.straighten:
                sv, size, _ = engine._target(an, 900)
                g = geometry.auto_angle(sv, dict(engine._masks_at(an, size)))
                if abs(g["angle"]) >= 0.4 and g["confidence"] >= 0.4:
                    params.geometry = Geometry(enabled=True, angle=float(g["angle"]))
                    note = f", levelled {g['angle']:+.1f}°"
            path, size = exporter.export(an, params, a.preset, str(out / stem))
            an.cache.clear()
            flag = f" [{q.get('blur')}]" if q.get("blur") in ("blurry", "soft") else ""
            print(f"[{n}/{len(files)}] {f.name} -> {Path(path).name}  ({lid}{note}){flag}", flush=True)
            done += 1
        except Exception as e:
            failed += 1
            print(f"[{n}/{len(files)}] {f.name}: FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
            traceback.print_exc(limit=1)
    print(f"finished in {(time.time() - t0) / 60:.1f} min: {done} edited, {skipped} skipped, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
