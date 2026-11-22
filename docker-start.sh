#!/usr/bin/env bash
# =============================================================================
# LPU Campus Parking Intelligence System - Docker Launcher for Linux / macOS
# =============================================================================

set -e

if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed or not in PATH."
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

echo "[INFO] Building and launching containerized application..."
docker compose up --build -d

echo ""
echo "========================================================================="
echo " LPU Campus Parking Intelligence System is running in Docker!"
echo "========================================================================="
echo " Web Interface:    http://localhost:8000"
echo " API Healthcheck:  http://localhost:8000/api/health"
echo " OpenAPI Docs:     http://localhost:8000/docs"
echo ""
echo " Sign in as 'admin'. Password = \$ADMIN_PASSWORD if you set it, otherwise the"
echo " one-time password printed by:  docker compose logs | grep 'one-time'"
echo ""
echo " To view container logs:  docker compose logs -f"
echo " To stop container:       docker compose down"
echo "========================================================================="
