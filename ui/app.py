#!/usr/bin/env python3
"""
app.py — Server Antarmuka Web Modern (Full Dashboard) untuk
Seeed Grove Vision AI V2 (DOC Sexing PP2 - MobileNetV2 INT8 on Arm Ethos-U55 NPU)
Mendukung Dual Mode: Live Webcam Camera & Input Gambar / Preset Dataset.
"""

import os
import sys
import json
import glob
import time
import base64
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import io
from PIL import Image

# Import modul serial lokal
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.insert(0, CURRENT_DIR)

import grove_serial

PORT_HTTP = 8080
grove_client = grove_serial.GroveVisionClient()


def get_preset_samples():
    """Mengumpulkan sampel citra representatif yang jelas (fokus bulu sayap DOC) untuk pengujian cepat."""
    samples = []
    female_dir = os.path.join(PROJECT_ROOT, "PP2/data/doc_augmented/test/Female")
    male_dir   = os.path.join(PROJECT_ROOT, "PP2/data/doc_augmented/test/Male")
    
    # Citra pilihan berkualitas tinggi dengan fokus bulu sayap yang jelas (bukan tangan/potongan ubin)
    curated_females = [
        os.path.join(female_dir, "female_test_0086.jpg"),
        os.path.join(female_dir, "female_test_0049.jpg"),
        os.path.join(female_dir, "female_test_0088.jpg"),
    ]
    curated_males = [
        os.path.join(male_dir, "male_test_0026.jpg"),
        os.path.join(male_dir, "male_test_0069.jpg"),
        os.path.join(male_dir, "male_test_0075.jpg"),
    ]

    females = [f for f in curated_females if os.path.exists(f)]
    if len(females) < 3 and os.path.exists(female_dir):
        for f in sorted(glob.glob(os.path.join(female_dir, "*.*"))):
            if f not in females:
                females.append(f)
            if len(females) >= 3:
                break

    males = [m for m in curated_males if os.path.exists(m)]
    if len(males) < 3 and os.path.exists(male_dir):
        for m in sorted(glob.glob(os.path.join(male_dir, "*.*"))):
            if m not in males:
                males.append(m)
            if len(males) >= 3:
                break

    for idx, f in enumerate(females[:3]):
        samples.append({
            "id": f"female_{idx+1}",
            "name": f"Betina (Female) #{idx+1}",
            "expected": "BETINA (FEMALE)",
            "badge": "female",
            "path": f
        })
    for idx, m in enumerate(males[:3]):
        samples.append({
            "id": f"male_{idx+1}",
            "name": f"Jantan (Male) #{idx+1}",
            "expected": "JANTAN (MALE)",
            "badge": "male",
            "path": m
        })

    # Tambahkan opsi Blank / Sintetis
    samples.append({
        "id": "blank_frame",
        "name": "Blank (Tanpa Objek)",
        "expected": "BLANK / AMBIGU",
        "badge": "ambiguous",
        "path": "SYNTHETIC:BLANK"
    })
    return samples


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler untuk menyajikan Dashboard dan REST API."""

    def log_message(self, format, *args):
        # Mute logging static file requests agar terminal tetap bersih
        if self.path.startswith("/static/"):
            return
        super().log_message(format, *args)

    def send_json_response(self, data, status_code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # 1. Halaman Utama
        if path == "/" or path == "/index.html":
            html_path = os.path.join(CURRENT_DIR, "templates/index.html")
            if os.path.exists(html_path):
                with open(html_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "File templates/index.html tidak ditemukan")
            return

        # 2. File Statis (CSS, JS, Assets)
        if path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            file_path = os.path.join(CURRENT_DIR, "static", rel_path)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                content_type = "text/plain"
                if file_path.endswith(".css"):
                    content_type = "text/css"
                elif file_path.endswith(".js"):
                    content_type = "application/javascript"
                elif file_path.endswith(".png"):
                    content_type = "image/png"
                elif file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
                    content_type = "image/jpeg"
                elif file_path.endswith(".svg"):
                    content_type = "image/svg+xml"

                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, f"File static tidak ditemukan: {path}")
            return

        # 3. API: Daftar Serial Ports
        if path == "/api/ports":
            ports = grove_serial.list_available_ports()
            self.send_json_response({"success": True, "ports": ports})
            return

        # 4. API: Daftar Preset Samples
        if path == "/api/samples":
            samples = get_preset_samples()
            self.send_json_response({"success": True, "samples": samples})
            return

        # 5. API: Thumbnail Sample Image
        if path == "/api/sample_image":
            qs = urllib.parse.parse_qs(parsed.query)
            sample_path = qs.get("path", [None])[0]
            if not sample_path:
                self.send_error(400, "Parameter 'path' dibutuhkan")
                return

            if sample_path == "SYNTHETIC:BLANK":
                blank = Image.new("RGB", (160, 160), color=(128, 128, 128))
                buf = io.BytesIO()
                blank.save(buf, format="JPEG")
                img_bytes = buf.getvalue()
            elif os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    img_bytes = f.read()
            else:
                self.send_error(404, "File sampel tidak ditemukan di disk")
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(img_bytes)))
            self.end_headers()
            self.wfile.write(img_bytes)
            return

        self.send_error(404, f"Endpoint tidak ditemukan: {path}")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b""

        # 1. API: Ping Hardware Grove
        if path == "/api/ping":
            data = json.loads(body.decode("utf-8")) if body else {}
            port = data.get("port")
            result = grove_client.ping(port=port)
            self.send_json_response(result)
            return

        # 2. API: Inferensi Citra (Kamera atau File)
        if path == "/api/infer":
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_json_response({"success": False, "error": f"JSON parse error: {str(e)}"}, 400)
                return

            port = data.get("port")
            img_b64 = data.get("image")        # Base64 data URL dari webcam/upload
            sample_path = data.get("sample_path") # Atau path ke sampel dataset

            pil_img = None
            if sample_path:
                if sample_path == "SYNTHETIC:BLANK":
                    pil_img = Image.new("RGB", (160, 160), color=(128, 128, 128))
                elif os.path.exists(sample_path):
                    pil_img = Image.open(sample_path).convert("RGB")
                else:
                    self.send_json_response({"success": False, "error": f"File sampel tidak ditemukan: {sample_path}"}, 404)
                    return
            elif img_b64:
                # Format: "data:image/jpeg;base64,...."
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(img_b64)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                self.send_json_response({"success": False, "error": "Tidak ada data gambar yang dikirim"}, 400)
                return

            # Preprocessing ke 160x160 untuk preview & kirim
            pil_160, rgb_arr, raw_bytes, checksum = grove_serial.preprocess_image_to_rgb160(pil_img)
            
            # Buat base64 dari citra 160x160 yang diproses agar user bisa verifikasi
            buf = io.BytesIO()
            pil_160.save(buf, format="JPEG", quality=90)
            preprocessed_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

            # Kirim ke modul Grove Vision AI V2
            infer_result = grove_client.infer(pil_160, port=port)
            infer_result["preprocessed_image"] = preprocessed_b64
            
            self.send_json_response(infer_result)
            return

        self.send_error(404, f"API POST endpoint tidak ditemukan: {path}")


def run_server(port=PORT_HTTP):
    ThreadingHTTPServer.allow_reuse_address = True
    server_address = ("0.0.0.0", port)
    httpd = ThreadingHTTPServer(server_address, DashboardHandler)
    print("=" * 70)
    print("  🐔 Seeed Grove Vision AI V2 — DOC Sexing AI Dashboard")
    print("=" * 70)
    print(f"  Aplikasi UI Web aktif di:")
    print(f"  👉 http://localhost:{port}")
    print(f"  👉 http://127.0.0.1:{port}")
    print("=" * 70)
    print("  [Fitur Dashboard]")
    print("  * Mode 1: Live Webcam (Kamera PC/Laptop/USB via browser)")
    print("  * Mode 2: Input Gambar (Upload File & Preset Sampel Female/Male)")
    print("  * Protokol Serial: 'PP2IR' High-Speed @ 921600 Baud")
    print("  * Eksekusi NPU on-device Arm Ethos-U55 (~3 ms) & Output I2C ke Arduino")
    print("=" * 70)
    print("Tekan [Ctrl + C] untuk menghentikan server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Menghentikan server UI...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    p = PORT_HTTP
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        p = int(sys.argv[1])
    run_server(port=p)

