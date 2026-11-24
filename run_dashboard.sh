#!/usr/bin/env bash
# ========================================================
# AGRI-06 Smart Hatchery — PP2 DOC Sexing AI Dashboard
# One-Click Launcher untuk Linux & macOS
# ========================================================

set -e

# Pindah ke direktori root proyek
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "  AGRI-06 Smart Hatchery — PP2 DOC Sexing AI Dashboard  "
echo "========================================================"
echo ""

# 1. Periksa ketersediaan Python 3
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 tidak ditemukan di sistem ini!"
    echo "Silakan install python3 dan python3-venv melalui package manager."
    exit 1
fi

# 2. Setup Virtual Environment Otomatis (.venv)
if [ ! -d ".venv" ]; then
    echo "[INFO] Menyiapkan environment Python terisolasi (.venv)..."
    python3 -m venv .venv
fi

# 3. Aktifkan Virtual Environment & Install Dependensi
source .venv/bin/activate
echo "[INFO] Memeriksa paket dependensi (pyserial, pillow, numpy)..."
pip install -r PP2/ui/requirements.txt --quiet --disable-pip-version-check

# 4. Buka Browser Otomatis
echo "[INFO] Membuka Dashboard Web di browser..."
if command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:8080" &>/dev/null &
elif command -v open &>/dev/null; then
    open "http://localhost:8080" &>/dev/null &
fi

# 5. Jalankan Server Dashboard Python
echo "[INFO] Menjalankan Server Web di http://localhost:8080"
echo "Tekan Ctrl+C untuk menghentikan server."
echo "========================================================"
python3 PP2/ui/app.py
