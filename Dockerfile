# =============================================================================
# LPU Campus Parking Intelligence System - Dockerfile (Render Free Tier)
#
# Optimized for Render Free Tier (512 MB RAM limit) and Docker Hub deployment:
#   docker build --platform linux/amd64 -t ali1131/campus-parking:latest .
#
# Runtime settings (environment variables):
#   PORT              Port to listen on (Render dynamically injects this, default 10000)
#   AUTH_DB           Accounts / calendar / audit database (default /app/data/live/app.db)
#   PARKING_DB        Camera count observations (default /app/data/live/observations.db)
#   ADMIN_USERNAME    First admin account (default: admin)
#   ADMIN_PASSWORD    First admin password (12+ chars, 1 digit, not containing 'admin')
#   COOKIE_SECURE=1   Behind HTTPS (Render standard)
#   INGEST_API_KEY    Key camera devices send in X-API-Key header
# =============================================================================
ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim-bookworm

# Python and system environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=10000 \
    FORWARDED_ALLOW_IPS=*

# Minimal OS runtime: libgomp1 for XGBoost OpenMP runtime, curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Unprivileged application user & writable directories
RUN groupadd -r -g 1000 appgroup \
    && useradd -r -u 1000 -g appgroup -d /home/appuser -m -s /bin/bash appuser \
    && mkdir -p /tmp/matplotlib /app/data/live /var/data \
    && chown -R appuser:appgroup /tmp/matplotlib /app /var/data

# Install ONLY lightweight runtime requirements (CPU-only, no PyTorch / OpenCV / Ultralytics)
COPY requirements-render.txt ./
RUN pip install --no-cache-dir -r requirements-render.txt

# Copy required application modules and frontend assets
COPY --chown=appuser:appgroup api/ ./api/
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup data_collection/ ./data_collection/
COPY --chown=appuser:appgroup features/ ./features/
COPY --chown=appuser:appgroup frontend/ ./frontend/
COPY --chown=appuser:appgroup public/ ./public/
COPY --chown=appuser:appgroup recommendation/ ./recommendation/
COPY --chown=appuser:appgroup utils/ ./utils/
COPY --chown=appuser:appgroup models/*.py ./models/
COPY --chown=appuser:appgroup artifacts/pipelines/ ./artifacts/pipelines/

# Setup container entrypoint
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && chmod +x /usr/local/bin/docker-entrypoint.sh

# Render binds to $PORT (default 10000)
EXPOSE 10000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-10000}/api/health" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]

# Single worker keeps memory at ~195 MB (safely below Render's 512 MB Free Tier limit)
CMD ["sh", "-c", "exec python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
