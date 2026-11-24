@echo off
TITLE AGRI-06 — Dashboard AI DOC Sexing
COLOR 0A

echo ========================================================
echo   AGRI-06 Smart Hatchery — PP2 DOC Sexing AI Dashboard
echo ========================================================
echo.

:: 1. Periksa ketersediaan Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python tidak ditemukan di sistem ini!
    echo Silakan install Python 3.9 - 3.12 dari https://www.python.org/
    echo Pastikan mencentang "Add Python to PATH" saat menginstall.
    echo.
    pause
    exit /b 1
)

:: 2. Navigasi ke direktori skrip ini
cd /d "%~dp0"

:: 3. Setup Virtual Environment Otomatis (.venv)
if not exist ".venv" (
    echo [INFO] Menyiapkan environment Python terisolasi (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Gagal membuat virtual environment.
        pause
        exit /b 1
    )
)

:: 4. Aktifkan Virtual Environment & Install Dependensi Ringan
call .venv\Scripts\activate.bat
echo [INFO] Memeriksa paket dependensi (pyserial, pillow, numpy)...
pip install -r PP2\ui\requirements.txt --quiet --disable-pip-version-check

:: 5. Buka Browser Otomatis
echo [INFO] Membuka Dashboard Web di browser...
start http://localhost:8080

:: 6. Jalankan Server Dashboard Python
echo [INFO] Menjalankan Server Web (Tekan Ctrl+C untuk berhenti)...
echo ========================================================
python PP2\ui\app.py

pause
