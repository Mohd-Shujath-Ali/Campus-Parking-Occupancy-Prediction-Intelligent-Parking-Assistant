# =============================================================================
# LPU Campus Parking Intelligence System - Dockerfile
#
#   docker build -t lpu-parking .                      # lean image (default): API, dashboards, forecasts,
#                                                      #   in-browser YOLO. ~0.4 GB RAM at runtime. Use this on Render.
#   docker build --target full -t lpu-parking:full .   # adds OpenCV + PyTorch + Ultralytics + the YOLO weights
#                                                      #   (server-side /api/detect-*). Needs ~2-3 GB RAM.
#
# Runtime settings (environment variables), all optional:
#   PORT            port to listen on (Render sets it; default 8000)
#   AUTH_DB         accounts / calendar / audit database   (default /app/data/live/app.db)
#   PARKING_DB      camera count observations              (default /app/data/live/observations.db)
#   ADMIN_USERNAME / ADMIN_PASSWORD   first admin account (password: 12+ chars, not containing the user ID)
#   COOKIE_SECURE=1 behind HTTPS (Render, any real deployment)
#   INGEST_API_KEY  key camera devices send in the X-API-Key header
# Put AUTH_DB / PARKING_DB on a mounted volume (Render disk, Docker volume) or accounts are lost on redeploy.
# =============================================================================
ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# The saved models were built and tested with Python 3.13 and the pinned versions in requirements.txt.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8000 \
    FORWARDED_ALLOW_IPS=*

# libgomp: OpenMP runtime XGBoost needs. curl: health check. (setpriv, used by the entrypoint, ships with util-linux.)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Unprivileged user the app runs as (the entrypoint starts as root only to fix volume ownership, then drops to this user)
RUN groupadd -r -g 1000 appgroup \
    && useradd -r -u 1000 -g appgroup -d /home/appuser -m -s /bin/bash appuser \
    && mkdir -p /tmp/matplotlib /app/data/live /var/data \
    && chown -R appuser:appgroup /tmp/matplotlib /app /var/data

# Dependencies first so code changes do not reinstall them. requirements.txt is the lean, pinned runtime set.
COPY requirements.txt requirements-full.txt ./
RUN pip install -r requirements.txt

# Application code, configs, trained models and assets. models/*.py only: the 100 MB YOLO weights belong to the full target.
COPY --chown=appuser:appgroup api/ ./api/
COPY --chown=appuser:appgroup artifacts/ ./artifacts/
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup computer_vision/ ./computer_vision/
COPY --chown=appuser:appgroup data/ ./data/
COPY --chown=appuser:appgroup data_collection/ ./data_collection/
COPY --chown=appuser:appgroup data_pipeline/ ./data_pipeline/
COPY --chown=appuser:appgroup evaluation/ ./evaluation/
COPY --chown=appuser:appgroup features/ ./features/
COPY --chown=appuser:appgroup frontend/ ./frontend/
COPY --chown=appuser:appgroup public/ ./public/
COPY --chown=appuser:appgroup recommendation/ ./recommendation/
COPY --chown=appuser:appgroup scripts/ ./scripts/
COPY --chown=appuser:appgroup utils/ ./utils/
COPY --chown=appuser:appgroup models/*.py ./models/

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# Strip Windows line endings (a CRLF script fails with "bad interpreter") and make it executable
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

# start-period covers the first import of pandas/XGBoost and the model load
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/api/health" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
# One worker: forecast history and rate limits live in process memory. --proxy-headers (uvicorn's default) plus
# FORWARDED_ALLOW_IPS=* make the real client address visible behind Render's proxy, so login rate limits are per person.
CMD ["sh", "-c", "exec python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

# -----------------------------------------------------------------------------
# Optional: server-side YOLO (/api/detect-image, /api/detect-frame)
# -----------------------------------------------------------------------------
FROM base AS full
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# CPU PyTorch wheel first (avoids ~2.5 GB of CUDA packages), then the rest
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-full.txt
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor \
    TORCH_HOME=/tmp/torch \
    YOLO_CONFIG_DIR=/tmp/Ultralytics
RUN mkdir -p /tmp/torch_inductor /tmp/torch /tmp/Ultralytics && chown -R appuser:appgroup /tmp/torch_inductor /tmp/torch /tmp/Ultralytics
COPY --chown=appuser:appgroup models/weights/ ./models/weights/

# -----------------------------------------------------------------------------
# Default target: the lean image. (Last stage = what `docker build` and Render build.)
# -----------------------------------------------------------------------------
FROM base AS lean
