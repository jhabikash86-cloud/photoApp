#!/bin/bash
# Stops the editor's servers (your edits are autosaved; albums are in library/library.db and are not affected).
for port in 8000 3000; do
  pids=$(lsof -ti tcp:$port -sTCP:LISTEN 2>/dev/null)
  [ -n "$pids" ] && kill $pids
done
echo "Neural Edit Pro stopped."
sleep 1
