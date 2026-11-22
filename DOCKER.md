# Docker Deployment Guide

## LPU Campus Parking Intelligence System

How to build, run and deploy the containerized app. For Render specifically, see [Deploying on Render](#6-deploying-on-render).

---

## 1. Prerequisites

- **Docker** 20.10+ or [Docker Desktop](https://www.docker.com/products/docker-desktop/), and **Docker Compose** V2 (`docker compose`).
- Resources for the default (lean) image: about **0.5 GB RAM** at runtime (measured 310 MB resident, 370 MB peak with all models loaded) and ~1 GB disk. The optional `full` image (server-side YOLO) needs 2-3 GB RAM and ~4 GB disk.

## 2. Two images from one Dockerfile

| Target | Contains | Use it for |
|---|---|---|
| **`lean`** (default) | API, dashboards, forecasts, calendar, login, in-browser YOLO (ONNX) | Render, any normal deployment |
| `full` | Lean + OpenCV, PyTorch (CPU), Ultralytics, the 100 MB YOLO weights | Server-side `/api/detect-image` and `/api/detect-frame` |

```bash
docker build -t lpu-parking .                          # lean
docker build --target full -t lpu-parking:full .       # full
```

## 3. Quick start with Docker Compose

```bash
export ADMIN_PASSWORD='Roadside-Pass-2026'    # 12+ chars, a letter and a digit, must not contain "admin"
docker compose up --build -d
docker compose logs -f
docker compose down                           # stop (data is kept); add -v to also delete the data volume
```

(PowerShell: `$env:ADMIN_PASSWORD='...'`. You can also put it in a `.env` file next to `docker-compose.yml`.)

Open http://localhost:8000 and sign in as `admin`. You are asked to change the password at first login. Without `ADMIN_PASSWORD`, a
one-time admin password is printed in `docker compose logs`.

Accounts, the campus calendar and camera counts live in the `parking-data` volume, so they survive rebuilds.
Demo accounts for trying every role: `docker compose exec parking-assistant python -m scripts.seed_demo_users --known-password "Demo-Park-2026"`
(the one-click sign-in buttons on the login page appear only on localhost).

## 4. Standalone Docker commands

```bash
docker build -t campus-parking-intelligence:latest .

docker run -d --name campus-parking-app -p 8000:8000 \
  -v parking-data:/data -e AUTH_DB=/data/app.db -e PARKING_DB=/data/observations.db \
  -e ADMIN_PASSWORD='Roadside-Pass-2026' \
  --restart unless-stopped campus-parking-intelligence:latest

curl http://localhost:8000/api/health
docker logs -f campus-parking-app
```

## 5. Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | Port to listen on (Render sets this itself) |
| `AUTH_DB` | `/app/data/live/app.db` | Accounts, sessions, calendar, audit log. **Put it on a volume.** |
| `PARKING_DB` | `/app/data/live/observations.db` | Camera counts |
| `ADMIN_USERNAME`, `ADMIN_PASSWORD` | `admin`, generated | First admin account (created only when no accounts exist) |
| `COOKIE_SECURE` | `0` | `1` when served over HTTPS |
| `INGEST_API_KEY` | unset | Key camera devices send as `X-API-Key` to `POST /api/ingest` |
| `FORWARDED_ALLOW_IPS` | `*` in the image, `127.0.0.1` in compose | Trust proxy headers (real client IP) from these addresses. Keep `*` only behind a proxy such as Render's. |
| `DEMO_LOGINS` | `auto` | One-click demo sign-in buttons: `auto` (this machine only), `on`, `off` |

## 6. Deploying on Render

The repo's `render.yaml` is a Blueprint for one Docker web service with a persistent disk:

1. Push the repo to GitHub (branch `main`).
2. Render dashboard → **New → Blueprint** → select the repo. Enter `ADMIN_PASSWORD` when asked. (Or **New → Web Service → Docker** and set the same variables by hand.)
3. Render builds the lean image and starts it. Health check: `/api/health`.
4. Open the service URL, sign in as `admin`, change the password, add users on the **Users** page.

Things to know:

- **Persistence needs a disk, and disks need a paid plan.** The blueprint uses `starter` + a 1 GB disk at `/var/data`, with `AUTH_DB` and `PARKING_DB` pointing there. On the free plan (no disks) accounts, the calendar and counts reset on every restart or redeploy and the service sleeps when idle: fine for a demo only. The blueprint explains how to switch to it.
- **One instance only.** A disk can't be shared, and forecast history and rate limits live in the process.
- **Cost / memory:** `starter` has 512 MB; the app peaks around 370 MB. Use a bigger plan if you see out-of-memory restarts or build the `full` target.
- **Use an existing image instead of building on Render:** push the image to a registry and choose **New → Web Service → Existing image**; set the same environment variables and add the disk at `/var/data`.
- **Backups:** Render does not back up disk contents for you. Copy `/var/data/app.db` (use Render's shell or a scheduled job) if the accounts matter.
- **Ephemeral startup permissions:** the container starts as root only to give the unprivileged app user ownership of the database folders (the disk is mounted root-owned), then drops to that user.

## 7. Development mode (hot reload)

```bash
docker compose -f docker-compose.dev.yml up --build
```

Mounts `./api`, `./frontend`, `./public`, `./models`, `./features`, `./recommendation`, `./computer_vision`, `./configs` and `./artifacts` into the container; uvicorn restarts on change.

## 8. GPU (optional)

`Dockerfile.gpu` (CUDA 12.1, server-side YOLO on an NVIDIA GPU, needs the NVIDIA Container Toolkit). It is not built by any automated check; verify it on your GPU host.

```bash
docker build -f Dockerfile.gpu -t campus-parking-intelligence:gpu .
docker run -d --gpus all -p 8000:8000 -v parking-data:/data -e AUTH_DB=/data/app.db -e PARKING_DB=/data/observations.db \
  campus-parking-intelligence:gpu
```

## 9. Security and design notes

1. **Unprivileged runtime.** The app runs as `appuser` (UID 1000); `docker-entrypoint.sh` starts as root only to chown the database folders, then drops privileges with `setpriv`.
2. **No secrets in images.** `.dockerignore` excludes `data/live/` (accounts database, demo passwords, the admin one-time password), `*.db`, and `.env*`.
3. **Pinned lean dependencies.** `requirements.txt` pins the exact versions the saved models were built with; Python 3.13 matches.
4. **Real client addresses behind a proxy.** With `FORWARDED_ALLOW_IPS=*`, login rate limits apply per person, not to Render's proxy as a whole.
5. **Browser-side video.** Webcam frames are processed in the browser (ONNX YOLO); the server needs no camera access.
