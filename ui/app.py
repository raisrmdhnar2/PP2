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
    female_dir = os.path.join(PROJECT_ROOT, "PP2/data/doc/Female")
    male_dir   = os.path.join(PROJECT_ROOT, "PP2/data/doc/Male")
    
    # Citra pilihan berkualitas tinggi dengan fokus bulu sayap yang jelas (bukan tangan/potongan ubin)
    # Jumlah tombol di UI akan persis mengikuti daftar ini (bebas ditambah atau dikurangi)
    curated_females = [
        os.path.join(female_dir, "female_test_0002.jpg"),
        os.path.join(female_dir, "female_test_0003.jpg"),
    ]
    curated_males = [
        os.path.join(male_dir, "male_test_0001.jpg"),
        os.path.join(male_dir, "male_test_0002.jpg"),
    ]

    # Ambil persis file yang terdaftar dan ada di disk
    females = [f for f in curated_females if os.path.exists(f)]
    males   = [m for m in curated_males if os.path.exists(m)]

    # Fallback darurat: Jika daftar di atas dikosongkan sama sekali, baru baca dari folder
    if not females and os.path.exists(female_dir):
        females = sorted(glob.glob(os.path.join(female_dir, "*.*")))[:2]
    if not males and os.path.exists(male_dir):
        males = sorted(glob.glob(os.path.join(male_dir, "*.*")))[:2]

    for idx, f in enumerate(females):
        samples.append({
            "id": f"female_{idx+1}",
            "name": f"Betina (Female) #{idx+1}",
            "expected": "BETINA (FEMALE)",
            "badge": "female",
            "path": f
        })
    for idx, m in enumerate(males):
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

        # 2. API: Pratinjau Preprocessing Citra (160x160 RGB)
        if path == "/api/preprocess_preview":
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_json_response({"success": False, "error": f"JSON parse error: {str(e)}"}, 400)
                return

            img_b64 = data.get("image")
            sample_path = data.get("sample_path")
            pil_img = None

            if sample_path:
                if sample_path == "SYNTHETIC:BLANK":
                    pil_img = Image.new("RGB", (160, 160), color=(128, 128, 128))
                elif os.path.exists(sample_path):
                    pil_img = Image.open(sample_path)
                else:
                    self.send_json_response({"success": False, "error": f"File tidak ditemukan: {sample_path}"}, 404)
                    return
            elif img_b64:
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(img_b64)
                pil_img = Image.open(io.BytesIO(img_bytes))
            else:
                self.send_json_response({"success": False, "error": "Tidak ada data gambar"}, 400)
                return

            orig_w, orig_h = pil_img.size
            pil_160, rgb_arr, raw_bytes, checksum = grove_serial.preprocess_image_to_rgb160(pil_img)
            quality_stats = grove_serial.compute_image_quality(rgb_arr)

            buf = io.BytesIO()
            pil_160.save(buf, format="JPEG", quality=92)
            preprocessed_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

            self.send_json_response({
                "success": True,
                "preprocessed_image": preprocessed_b64,
                "orig_size": [orig_w, orig_h],
                "target_size": [160, 160],
                "checksum": f"0x{checksum:04X}",
                "quality": quality_stats
            })
            return

        # 3. API: Inferensi Citra (Kamera atau File)
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
                    pil_img = Image.open(sample_path)
                else:
                    self.send_json_response({"success": False, "error": f"File sampel tidak ditemukan: {sample_path}"}, 404)
                    return
            elif img_b64:
                # Format: "data:image/jpeg;base64,...."
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(img_b64)
                pil_img = Image.open(io.BytesIO(img_bytes))
            else:
                self.send_json_response({"success": False, "error": "Tidak ada data gambar yang dikirim"}, 400)
                return

            # Preprocessing ke 160x160 untuk preview & kirim
            pil_160, rgb_arr, raw_bytes, checksum = grove_serial.preprocess_image_to_rgb160(pil_img)
            quality_stats = grove_serial.compute_image_quality(rgb_arr)
            
            # Buat base64 dari citra 160x160 yang diproses agar user bisa verifikasi
            buf = io.BytesIO()
            pil_160.save(buf, format="JPEG", quality=92)
            preprocessed_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

            # Kirim ke modul Grove Vision AI V2
            infer_result = grove_client.infer(pil_160, port=port)
            infer_result["preprocessed_image"] = preprocessed_b64
            infer_result["quality"] = quality_stats
            
            self.send_json_response(infer_result)
            return

        self.send_error(404, f"API POST endpoint tidak ditemukan: {path}")


def run_server(port=PORT_HTTP):
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = None
    active_port = port

    # Coba port target; jika port 8080 masih dipakai, coba port berikutnya (8081, dst.)
    for p_candidate in range(port, port + 10):
        try:
            server_address = ("0.0.0.0", p_candidate)
            httpd = ThreadingHTTPServer(server_address, DashboardHandler)
            active_port = p_candidate
            break
        except OSError as e:
            if e.errno == 98:  # Address already in use
                continue
            raise

    if httpd is None:
        print(f"[ERROR] Port {port} s/d {port+9} semuanya sedang digunakan!")
        print("Silakan matikan proses lama dengan: fuser -k 8080/tcp")
        return

    print("=" * 70)
    print("  🐔 Seeed Grove Vision AI V2 — DOC Sexing AI Dashboard")
    print("=" * 70)
    print(f"  Aplikasi UI Web aktif di:")
    print(f"  👉 http://localhost:{active_port}")
    print(f"  👉 http://127.0.0.1:{active_port}")
    if active_port != port:
        print(f"  💡 (Catatan: Port {port} sedang sibuk, otomatis dialihkan ke {active_port})")
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

