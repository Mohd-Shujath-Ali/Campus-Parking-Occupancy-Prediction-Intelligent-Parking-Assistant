@echo off
REM =============================================================================
REM LPU Campus Parking Intelligence System - Docker Launcher for Windows
REM =============================================================================

echo [INFO] Checking Docker installation...
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH.
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop/
    echo and ensure Docker Desktop is running before executing this script.
    pause
    exit /b 1
)

echo [INFO] Building and starting containerized application...
docker compose up --build -d

if %ERRORLEVEL% equ 0 (
    echo.
    echo =========================================================================
    echo  LPU Campus Parking Intelligence System is running in Docker!
    echo =========================================================================
    echo  Web Interface:    http://localhost:8000
    echo  API Healthcheck:  http://localhost:8000/api/health
    echo  OpenAPI Docs:     http://localhost:8000/docs
    echo.
    echo  Sign in as "admin". Password = ADMIN_PASSWORD if you set it, otherwise the
    echo  one-time password printed by:  docker compose logs ^| findstr "one-time"
    echo.
    echo  To view container logs:  docker compose logs -f
    echo  To stop container:       docker compose down
    echo =========================================================================
) else (
    echo [ERROR] Failed to start Docker container.
)
pause
