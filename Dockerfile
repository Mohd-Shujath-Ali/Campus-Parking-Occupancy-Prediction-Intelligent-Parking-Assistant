# =============================================================================
# LPU Campus Parking Intelligence System - Production Dockerfile
# Multi-stage optimized Debian Bookworm runtime with non-root security.
# =============================================================================

FROM python:3.11-slim-bookworm AS runner

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor \
    TORCH_HOME=/tmp/torch \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8000 \
    HOST=0.0.0.0

# Install minimal OS dependencies required by OpenCV (headless) and OpenMP (XGBoost/PyTorch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root system user for security compliance
RUN groupadd -r -g 1000 appgroup && \
    useradd -r -u 1000 -g appgroup -d /home/appuser -m -s /bin/bash appuser && \
    mkdir -p /tmp/torch_inductor /tmp/torch /tmp/Ultralytics /tmp/matplotlib && \
    chown -R appuser:appgroup /tmp/torch_inductor /tmp/torch /tmp/Ultralytics /tmp/matplotlib /app

# Copy dependency manifest
COPY --chown=appuser:appgroup requirements.txt requirements-full.txt ./

# Install CPU PyTorch wheel first to avoid pulling 2.5GB CUDA packages, then install rest
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-full.txt

# Copy application code, configs, pipelines and assets
COPY --chown=appuser:appgroup api/ ./api/
COPY --chown=appuser:appgroup artifacts/ ./artifacts/
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup computer_vision/ ./computer_vision/
COPY --chown=appuser:appgroup data/ ./data/
COPY --chown=appuser:appgroup data_pipeline/ ./data_pipeline/
COPY --chown=appuser:appgroup evaluation/ ./evaluation/
COPY --chown=appuser:appgroup features/ ./features/
COPY --chown=appuser:appgroup frontend/ ./frontend/
COPY --chown=appuser:appgroup public/ ./public/
COPY --chown=appuser:appgroup models/ ./models/
COPY --chown=appuser:appgroup recommendation/ ./recommendation/
COPY --chown=appuser:appgroup utils/ ./utils/
COPY --chown=appuser:appgroup data_collection/ ./data_collection/

# Switch to non-root user
USER appuser

# Expose web service port
EXPOSE 8000

# Health check verifies FastAPI status and model availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Launch ASGI server
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
