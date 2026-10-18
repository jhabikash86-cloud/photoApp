#!/bin/bash
# Starts the editor: backend on :8000 and frontend on :3000. Stop both with Ctrl+C.
cd "$(dirname "$0")"
trap 'kill 0' EXIT
./venv/bin/uvicorn server:app --port 8000 &
(cd pro-editor-frontend && npm run dev) &
wait
