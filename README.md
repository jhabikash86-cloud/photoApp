# Neural Edit Pro

Local, free (no per-photo API cost) photo editor: FastAPI backend (`server.py`) + Next.js frontend (`pro-editor-frontend`).

## Setup
1. `python3 -m venv venv && ./venv/bin/pip install -r requirements.txt`
2. `cd pro-editor-frontend && npm install`
3. Copy `.env.example` to `.env` and fill in IMMICH_URL / IMMICH_API_KEY if you use Immich batch.
4. Put the models in `models/` (not in git, ~220 MB): `face_landmarker.task`, `selfie_multiclass.tflite`,
   `segformer_b2_ade.onnx` + `segformer_ade_config.json`, `depth_anything_v2_small.onnx` + `_preprocessor.json`.
5. `./start.sh` (backend :8000, frontend :3000). Tests: `./venv/bin/python image_quality_tests.py <photos>`.

## Sky library (optional)
`./venv/bin/python fetch_skies.py` downloads the free CC0 "Pure Skies" panoramas from Poly Haven (59 files, ~1 GB) into `skies/` (not in git).
They power Sky tab -> "Replace the sky". Safe to re-run; finished files are skipped.

## Using it like an app
- Double-click **Neural Edit Pro.app** (drag it to the Dock or Applications). It starts the two servers if they are not running and opens the editor in its own window (Chrome app mode). The servers keep running in the background, so reopening is instant.
- **Stop Neural Edit Pro.command** stops the servers. Logs: `~/Library/Logs/NeuralEditPro/`.
- For a dedicated Dock icon, open the editor in Chrome and choose menu -> Cast, save and share -> Install page as app.
- After changing the interface code, rebuild its production bundle: `cd pro-editor-frontend && NEP_PROD=1 npm run build` (the dev server `npm run dev` is separate).
- Your edits autosave to `edits/`; the photo library catalog lives in `library/`. Back both up with `./venv/bin/python backup_library.py backup <folder>`.
