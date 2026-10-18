# Neural Edit Pro

Local, free (no per-photo API cost) photo editor: FastAPI backend (`server.py`) + Next.js frontend (`pro-editor-frontend`).

## Setup
1. `python3 -m venv venv && ./venv/bin/pip install -r requirements.txt`
2. `cd pro-editor-frontend && npm install`
3. Copy `.env.example` to `.env` and fill in IMMICH_URL / IMMICH_API_KEY if you use Immich batch.
4. Put the models in `models/` (not in git, ~220 MB): `face_landmarker.task`, `selfie_multiclass.tflite`,
   `segformer_b2_ade.onnx` + `segformer_ade_config.json`, `depth_anything_v2_small.onnx` + `_preprocessor.json`.
5. `./start.sh` (backend :8000, frontend :3000). Tests: `./venv/bin/python image_quality_tests.py <photos>`.
