@echo off
REM Quick start script for DrugCLIP Web Application (Windows)

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║      DrugCLIP Web Application - Quick Start                  ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

echo [1/5] Initializing project structure...
python web_app/setup.py

echo.
echo [2/5] Installing dependencies...
cd web_app/backend
pip install -r requirements.txt -q
cd ../..

echo.
echo [3/5] Checking CUDA availability...
python -c "import torch; print('✓ CUDA available' if torch.cuda.is_available() else '⚠ CUDA not available, using CPU')"

echo.
echo ✓ Setup complete!
echo.
echo Next steps:
echo.
echo Option 1: Local Development
echo   1. Download model and data (see README.md)
echo   2. Run: cd web_app\backend ^& python app.py
echo   3. Open: http://localhost:5000
echo.
echo Option 2: Docker
echo   1. Install Docker Desktop
echo   2. Run: docker-compose -f web_app/docker-compose.yml up
echo   3. Open: http://localhost:5000
echo.
echo Documentation: see web_app/README.md
echo.
pause
