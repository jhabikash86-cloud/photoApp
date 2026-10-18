"""Unattended finish: wait for the NAS copy, upload to Immich, build the albums, then verify the NAS copy.

    ./venv/bin/python pipeline_after_copy.py <copy-log-file> <copy-process-pid>

Order matters on a Wi-Fi laptop: the copy and the upload would fight for the same bandwidth, so they run one after the other, and the
read-back verification (which re-reads 260 GB from the NAS) goes last. Progress is written to library/pipeline.log.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = str(HERE / "venv" / "bin" / "python")
LOG = HERE / "library" / "pipeline.log"
log_file, copy_pid = sys.argv[1], int(sys.argv[2])


def say(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


def run(args, name):
    say(f"START {name}")
    with open(LOG, "a") as f:
        rc = subprocess.call([PY] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    say(f"END {name} (exit {rc})")
    return rc


say("waiting for the NAS copy to finish")
while True:
    try:
        if "copy done" in Path(log_file).read_text():
            break
    except OSError:
        pass
    try:
        os.kill(copy_pid, 0)
    except OSError:
        say("copy process ended without 'copy done' (see its log); stopping")
        sys.exit(1)
    time.sleep(30)
try:
    os.kill(copy_pid, signal.SIGTERM)             # it would start verifying straight away; verification runs last
except OSError:
    pass
time.sleep(3)
run(["mirror_library.py", "status"], "copy status")
for attempt in range(3):                          # a resumable upload: run again for anything that failed
    run(["immich_push.py", "upload"], f"upload to Immich (pass {attempt + 1})")
    todo = subprocess.run([PY, "-c", "import sqlite3;db=sqlite3.connect('library/library.db');print(db.execute('SELECT COUNT(*) FROM photos WHERE dup_of IS NULL AND id NOT IN (SELECT photo_id FROM immich_map)').fetchone()[0])"],
                          cwd=HERE, capture_output=True, text=True).stdout.strip()
    say(f"{todo} photos still not in Immich")
    if todo == "0":
        break
run(["immich_push.py", "albums"], "create Immich albums")
run(["immich_push.py", "status"], "immich status")
run(["mirror_library.py", "verify"], "verify the NAS copy (read-back)")
run(["mirror_library.py", "status"], "final copy status")
say("ALL DONE")
