# Hugging Face Space (Docker SDK) that builds the app straight from GitHub.
#
# Upload this file and README.md to a new Space (SDK: Docker). To deploy a different branch or a fork,
# change GIT_REF / REPO. Private repository: add a Space secret named GITHUB_TOKEN (a fine-grained token
# with read access to the repo); it is used only while building and is not kept in the image.
FROM python:3.13-slim

ARG REPO=github.com/vibin-123/Campus-Parking-Occupancy-Prediction-Intelligent-Parking-Assistant
ARG GIT_REF=testing

# git: fetch the code. libgomp1: required by the XGBoost wheel. libglib2.0-0: required by OpenCV.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libgomp1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Spaces run as a non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Clone the repository (the token secret, if provided, is read from a build secret mount and dropped with .git)
RUN --mount=type=secret,id=GITHUB_TOKEN,mode=0444,required=false \
    if [ -s /run/secrets/GITHUB_TOKEN ]; then URL="https://$(cat /run/secrets/GITHUB_TOKEN)@${REPO}.git"; \
    else URL="https://${REPO}.git"; fi && \
    git clone --depth 1 --branch "${GIT_REF}" "${URL}" . && rm -rf .git

# Lean, pinned runtime dependencies (no torch/ultralytics; the browser runs YOLO via ONNX)
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements-render.txt

# Writable location for collected observations (SQLite)
RUN mkdir -p data/live

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
