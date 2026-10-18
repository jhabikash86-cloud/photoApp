"""Unattended: upload the travel library (RAW + edited) to Immich from the original drive, build the "Travel ·" albums, refresh the catalog backup.

    ./venv/bin/python pipeline_immich.py        # progress: library/pipeline.log
Safe to stop and start again: photos already in Immich are skipped.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = str(HERE / "venv" / "bin" / "python")
LOG = HERE / "library" / "pipeline.log"


def say(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


def run(args, name):
    say(f"START {name}")
    with open(LOG, "a") as f:
        rc = subprocess.call([PY] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    say(f"END {name} (exit {rc})")
    return rc


for attempt in range(4):
    run(["immich_push.py", "upload"], f"upload to Immich (pass {attempt + 1})")
    todo = subprocess.run([PY, "-c", "import sqlite3;db=sqlite3.connect('library/library.db');print(db.execute('SELECT COUNT(*) FROM photos WHERE dup_of IS NULL AND id NOT IN (SELECT photo_id FROM immich_map)').fetchone()[0])"],
                          cwd=HERE, capture_output=True, text=True).stdout.strip()
    say(f"{todo} photos still not in Immich")
    if todo == "0":
        break
run(["immich_push.py", "albums"], "create Immich albums")
run(["immich_push.py", "status"], "immich status")
run(["backup_library.py", "backup"], "refresh the catalog backup")
say("ALL DONE")
