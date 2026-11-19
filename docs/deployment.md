# Deployment

The app is one FastAPI service: it serves the dashboards (`/`, `/admin`, `/analytics`, `/zones`, `/vision`),
the REST API (`/api/...`, docs at `/docs`) and the browser-side YOLO model from the same origin.

## Option A: Render (recommended, uses `render.yaml`)
1. Push the branch you want to deploy (the blueprint targets `testing`; edit `branch:` in `render.yaml` to change it).
2. In the Render dashboard: **New → Blueprint**, connect the GitHub repository, select the branch, **Apply**.
3. Wait for the build (installs `requirements-render.txt`), then open the `onrender.com` URL Render shows.
   Health check: `/api/health`.
4. Find the generated `INGEST_API_KEY` under the service's **Environment** tab. Send it as the `X-API-Key`
   header when posting camera counts to `/api/ingest`.

Behaviour to expect on the free plan: the service sleeps after ~15 minutes idle (the first request then
takes ~30-60 s), memory is ~400 MB of 512 MB, and the instance disk is ephemeral (collected observations
are lost on redeploy).

## Option B: Docker (any host)
```bash
docker compose up --build -d        # CPU image, serves on http://localhost:8000
```
`Dockerfile` installs the full `requirements.txt` (including torch/ultralytics), so server-side YOLO works.
See `DOCKER.md` for the GPU variant.

## What the slim deployment includes
| Feature | Hosted (`requirements-render.txt`) |
|---|---|
| Forecasts, intervals, saturation risk, recommendations, demand outlook, heatmaps, zone analysis | yes |
| `/api/explain` (SHAP breakdown, computed with XGBoost's native TreeSHAP) | yes |
| In-browser YOLO on `/vision` (ONNX, runs on the user's device) | yes |
| Server-side YOLO (`/api/detect-image`, `/api/detect-frame`) | no (needs `ultralytics`/`torch`; use Docker) |
| Retraining (`python -m models.train`) and experiments | no (install `requirements.txt` locally) |

## Why the pinned versions
`artifacts/pipelines/*.joblib` are pickles produced with scikit-learn 1.9.1, XGBoost 3.4.1, NumPy 2.5.3,
pandas 3.0.5 on Python 3.13. `requirements-render.txt` pins exactly these so they load without version
skew. If you retrain with different versions, regenerate the pickles in the same environment you deploy.
