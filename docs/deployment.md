# Deployment

The app is one FastAPI service: it serves the dashboards (`/`, `/admin`, `/analytics`, `/zones`, `/vision`)
and the REST API (`/api/...`, docs at `/docs`). YOLO runs **in the browser** (ONNX, WebGPU/WASM) on the user's
own GPU/CPU, so the server needs no vision libraries.

## Option A: Vercel (primary)

### How the repository is set up for it
| File | Purpose |
|---|---|
| `pyproject.toml` | `[tool.vercel] entrypoint = "api.main:app"` plus the lean, pinned dependencies |
| `uv.lock` | Locked install (35 packages; CPU-only XGBoost, no torch/OpenCV/CUDA) |
| `requirements.txt` | Same lean set (identical to `pyproject.toml`, so either manifest gives the same install) |
| `vercel.json` | Function settings: 60 s max duration; keeps tests/notebooks/weights/`public/` out of the function bundle |
| `.vercelignore` | Keeps the uploaded source small (drops the 346 MB YOLO dataset, notebooks, figures, weights, ...) |
| `public/` | Static assets (CSS/JS, logo, **ONNX YOLO models**) served from Vercel's CDN, not by the function |

Measured with the lockfile: dependencies 288 MB + project files 78 MB ≈ **394 MB** against Vercel's
[500 MB Python function limit](https://vercel.com/changelog/python-vercel-functions-bundle-size-limit-increased-to-500mb).
Running from that simulated bundle: cold start ≈ 2 s, ≈ 360 MB RAM, every page and API route returns 200.

### Deploy
1. Make sure the branch you deploy contains these files (merge `testing` into `main`, or choose the branch below).
2. Vercel dashboard → **Add New → Project** → import the GitHub repository.
3. Settings on the import screen: **Framework Preset: FastAPI**, **Root Directory: `./`**; leave the
   Build, Output and Install commands **empty** (Vercel installs from `pyproject.toml`/`uv.lock`).
4. Optional environment variable: `INGEST_API_KEY` (any secret string). If set, `POST /api/ingest` requires it
   as the `X-API-Key` header.
5. **Deploy.** Vercel deploys the *production branch* (usually `main`) and builds every other branch as a preview.
   To deploy `testing` as production, change **Settings → Git → Production Branch**.

### What behaves differently on Vercel (serverless)
- **No server-side YOLO.** `/api/detect-image` and `/api/detect-frame` return `503` with an explanation; the
  Vision page runs the ONNX model in the browser. (Docker with `requirements-full.txt` keeps the server fallback.)
- **No shared memory between requests.** Each instance keeps its own rolling history, so forecasts usually report
  `lag_source: "estimated"` unless the caller supplies `lag_counts`; the in-flight dispatch tracker and rate limits
  are per instance.
- **Collected counts are not durable.** `POST /api/ingest` writes SQLite to the instance's `/tmp`, which is
  discarded when the instance is recycled. Use a hosted database if you need to keep them.
- **Cold starts.** After idle time the first request takes a few seconds while Python imports and the models load.
- **Limits.** Request and response bodies are capped at ~4.5 MB, which is why the ONNX models live in `public/`.

## Option B: Render with Docker (`render.yaml`)
Blueprint for one Docker web service built from `./Dockerfile` (the lean image) on the `main` branch, with a 1 GB
persistent disk at `/var/data` for the SQLite databases. **Render dashboard → New → Blueprint → select the repo**, and
enter `ADMIN_PASSWORD` when asked. Step by step, plan/memory notes and the free-plan variant: [DOCKER.md](../DOCKER.md#6-deploying-on-render).

Important: accounts, the calendar and camera counts are SQLite files. They persist only on a disk (paid plans, one
instance); without one they reset on every restart. Set `COOKIE_SECURE=1` (the blueprint does).

## Option C: Docker locally
```bash
export ADMIN_PASSWORD='Roadside-Pass-2026'
docker compose up --build -d        # serves on http://localhost:8000
```
The default image is the lean one. `docker build --target full .` adds OpenCV, torch and Ultralytics for server-side
YOLO. See [DOCKER.md](../DOCKER.md) for the settings and the GPU variant.

## Which features work where
| Feature | Vercel / Render (lean) | Docker `full` target / local (full) |
|---|---|---|
| Login, accounts, calendar | Render with a disk: yes. Vercel: no (no persistent storage) | yes |
| Forecasts, intervals, saturation risk, recommendations, demand outlook, heatmaps, zone analysis | yes | yes |
| `/api/explain` (exact TreeSHAP via XGBoost) | yes | yes |
| In-browser YOLO (ONNX) | yes | yes |
| Server-side YOLO (`/api/detect-*`) | no (503) | yes |
| Retraining and experiments | no | yes (`requirements-full.txt`) |

## Why the pinned versions
`artifacts/pipelines/*.joblib` are pickles produced with scikit-learn 1.9.1, XGBoost 3.4.1, NumPy 2.5.3 and
pandas 3.0.5. The lean requirements pin exactly these so the models load without version skew. If you retrain with
different versions, regenerate the pickles in the environment you deploy.
