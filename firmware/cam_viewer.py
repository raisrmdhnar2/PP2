#!/usr/bin/env python3
"""
Real-time Camera Stream Viewer & Inference Monitor
Seeed Grove Vision AI V2 -> PP2 Egg Sexing (MobileNetV2 INT8 + Ethos-U55 NPU)
I2C Communication to Arduino Uno R3 (Slave 0x55)

Author: Groove-Vision-Ai-v2 Project
"""

import sys
import os
import time
import json
import base64
import argparse
import datetime
import numpy as np
import cv2

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("[ERROR] Module 'pyserial' belum terinstall.")
    print("        Silakan jalankan: pip install pyserial opencv-python numpy")
    sys.exit(1)


os.environ["QT_LOGGING_RULES"] = "*=false"
os.environ["QT_QPA_PLATFORM"] = "xcb"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Grove Vision AI V2 Real-time Camera & AI Monitor")
    parser.add_argument("port", nargs="?", default=None,
                        help="Port serial modul Grove Vision AI V2 (posisional, misal: /dev/ttyACM0)")
    parser.add_argument("--port", "-p", dest="port_flag", type=str, default=None,
                        help="Port serial modul Grove Vision AI V2 (opsional, misal: -p /dev/ttyACM0)")
    parser.add_argument("--baud", "-b", type=int, default=921600,
                        help="Baudrate serial (default: 921600)")
    parser.add_argument("--save-dir", "-s", type=str, default="PP2/captures",
                        help="Direktori penyimpanan snapshot gambar (default: PP2/captures)")
    return parser.parse_args()


def auto_detect_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if "grove" in desc or "vision" in desc or "seeed" in desc or "acm" in p.device.lower():
            return p.device
    if len(ports) > 0:
        return ports[0].device
    return "/dev/ttyACM0"


def draw_hud(frame, meta, stream_fps):
    """
    Menggambar Head-Up Display (HUD) informatif di atas frame kamera.
    """
    h, w = frame.shape[:2]
    # Resize tampilan agar nyaman dilihat di monitor (misal 640x640 atau 640x480)
    display_w = 640
    display_h = int(h * (display_w / float(w))) if w > 0 else 480
    resized = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

    # Buat panel HUD di bagian bawah & atas
    hud_top_h = 75
    hud_bot_h = 135
    canvas = np.zeros((display_h + hud_top_h + hud_bot_h, display_w, 3), dtype=np.uint8)

    # Background frame
    canvas[hud_top_h:hud_top_h + display_h, 0:display_w] = resized

    # Metadata extraction
    class_id   = meta.get("class_id", 2)
    label      = meta.get("label", "undefined")
    conf       = float(meta.get("confidence", 0.0))
    p_betina   = float(meta.get("p_betina", 0.0))
    p_jantan   = float(meta.get("p_jantan", 0.0))
    pixel_std  = float(meta.get("pixel_std", 0.0))
    reason     = meta.get("reason", "UNKNOWN")
    npu_ms     = float(meta.get("npu_ms", 0.0))
    npu_fps    = float(meta.get("npu_fps", 0.0))
    i2c_status = meta.get("i2c_status", "N/A")
    frame_num  = meta.get("frame", 0)

    # 1. Header Bar
    cv2.rectangle(canvas, (0, 0), (display_w, hud_top_h), (25, 25, 30), -1)
    cv2.putText(canvas, "GROVE VISION AI V2 - PP2 EGG SEXING", (16, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)
    
    perf_str = f"NPU: {npu_ms:.1f}ms ({npu_fps:.1f} FPS) | Stream: {stream_fps:.1f} FPS | Frame #{frame_num}"
    cv2.putText(canvas, perf_str, (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # 2. Main Classification Decision
    # Tentukan warna status:
    # Hijau = Betina (Class 0), Biru = Jantan (Class 1), Oranye/Kuning = Undefined/Ragu/Blank (Class 2)
    if class_id == 0:
        theme_color = (60, 220, 70)      # Bright Green (Betina)
        status_title = f"BETINA (HEN) - {conf:.1f}%"
    elif class_id == 1:
        theme_color = (255, 140, 30)     # Bright Cyan/Blue (Jantan - BGR)
        status_title = f"JANTAN (COCK) - {conf:.1f}%"
    else:
        if reason == "BLANK_FRAME":
            theme_color = (0, 165, 255)   # Amber (Blank)
            status_title = f"BLANK FRAME (Std={pixel_std:.1f} < 12.0)"
        else:
            theme_color = (50, 180, 255)  # Yellow (Low Confidence)
            status_title = f"RAGU-RAGU / LOW CONF ({conf:.1f}% < 80%)"

    # Border frame kamera dengan warna tema status
    cv2.rectangle(canvas, (0, hud_top_h), (display_w - 1, hud_top_h + display_h), theme_color, 2)

    # 3. Footer Bar
    cv2.rectangle(canvas, (0, display_h + hud_top_h), (display_w, canvas.shape[0]), (20, 20, 24), -1)

    # Title status utama
    cv2.putText(canvas, status_title, (16, display_h + hud_top_h + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, theme_color, 2, cv2.LINE_AA)

    # Confidence Bars (Betina vs Jantan)
    bar_x = 16
    bar_y_base = display_h + hud_top_h + 46
    bar_w = 260
    bar_h = 10

    # Bar Betina
    cv2.putText(canvas, f"Betina: {p_betina:.1f}%", (bar_x, bar_y_base + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bar_x + 95, bar_y_base + 3), (bar_x + 95 + bar_w, bar_y_base + 3 + bar_h), (50, 50, 50), -1)
    fill_b = int((p_betina / 100.0) * bar_w)
    cv2.rectangle(canvas, (bar_x + 95, bar_y_base + 3), (bar_x + 95 + fill_b, bar_y_base + 3 + bar_h), (60, 220, 70), -1)

    # Bar Jantan
    cv2.putText(canvas, f"Jantan: {p_jantan:.1f}%", (bar_x, bar_y_base + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bar_x + 95, bar_y_base + 23), (bar_x + 95 + bar_w, bar_y_base + 23 + bar_h), (50, 50, 50), -1)
    fill_j = int((p_jantan / 100.0) * bar_w)
    cv2.rectangle(canvas, (bar_x + 95, bar_y_base + 23), (bar_x + 95 + fill_j, bar_y_base + 23 + bar_h), (255, 140, 30), -1)

    # Garis ambang batas 80% pada bar
    thresh_x = bar_x + 95 + int(0.80 * bar_w)
    cv2.line(canvas, (thresh_x, bar_y_base), (thresh_x, bar_y_base + 36), (0, 0, 255), 1)

    # Status I2C Arduino & Paket
    i2c_color = (0, 255, 120) if "OK" in i2c_status else (0, 0, 255)
    conf_scaled = int(round(conf * 100))
    conf_int = conf_scaled // 100
    conf_dec = conf_scaled % 100
    crc = 0xAA ^ class_id ^ conf_int ^ conf_dec
    pkt_hex = f"[0xAA, {class_id}, {conf_int}, {conf_dec}, 0x{crc:02X}]"

    cv2.putText(canvas, f"I2C (Uno 0x55): {i2c_status}", (display_w - 240, display_h + hud_top_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, i2c_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Packet: {pkt_hex}", (display_w - 240, display_h + hud_top_h + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)

    # Key shortcuts guide
    cv2.putText(canvas, "[s] Snapshot | [Space] Pause | [q] Exit", (display_w - 240, display_h + hud_top_h + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1, cv2.LINE_AA)

    return canvas


def main():
    args = parse_arguments()
    port = args.port_flag or args.port
    if not port or not os.path.exists(port):
        auto_port = auto_detect_port()
        if not port:
            port = auto_port
        else:
            print(f"[WARN] Port '{port}' tidak ditemukan. Menggunakan port terdeteksi: '{auto_port}'")
            port = auto_port

    if not os.path.exists(port):
        print(f"[FATAL] Tidak ada perangkat serial Grove Vision AI V2 terdeteksi di '{port}'.")
        print("        Pastikan modul terhubung via kabel USB dan port serial aktif.")
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"===========================================================")
    print(f" Grove Vision AI V2 - Real-time Camera & Egg Sexing Monitor")
    print(f" Port      : {port}")
    print(f" Baudrate  : {args.baud}")
    print(f" Snapshot  : {args.save_dir}/")
    print(f"===========================================================")

    try:
        ser = serial.Serial(port, args.baud, timeout=1.0)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"[FATAL] Gagal membuka port serial '{port}': {e}")
        print("Pastikan kabel USB modul Grove Vision AI V2 terpasang dengan baik.")
        sys.exit(1)

    window_name = "Grove Vision AI V2 - Egg Sexing Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    stream_fps = 0.0
    frame_count = 0
    paused = False
    last_hud = None

    print("[INFO] Menunggu stream JSON dari modul...")
    print("[INFO] (Jika tidak muncul gambar, pastikan modul sudah diflash dengan 'python flash_firmware.py')")

    try:
        while True:
            if not paused:
                line = ser.readline()
                if not line:
                    # Refresh OpenCV event loop
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        break
                    continue

                line_str = line.decode("utf-8", errors="ignore").strip()

                # Cek baris SSCMA JSON: harus mengandung '"image":'
                if not ('"image":' in line_str and "{" in line_str):
                    if line_str:
                        print(f"[BOARD LOG] {line_str}")
                    continue

                try:
                    msg = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                data = msg.get("data", {})
                img_b64 = data.get("image", None)

                if not img_b64:
                    continue

                try:
                    img_bytes = base64.b64decode(img_b64)
                    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                except Exception as ex:
                    print(f"[WARN] Gagal decode JPEG frame: {ex}")
                    continue

                if frame is None:
                    continue

                # Hitung FPS laptop stream
                frame_count += 1
                curr_time = time.time()
                dt = curr_time - prev_time
                if dt >= 0.5:
                    stream_fps = frame_count / dt
                    frame_count = 0
                    prev_time = curr_time

                # Gambar HUD overlay
                hud_frame = draw_hud(frame, data, stream_fps)
                last_hud = hud_frame
                cv2.imshow(window_name, hud_frame)

            else:
                # Keadaan Paused
                if last_hud is not None:
                    pause_view = last_hud.copy()
                    cv2.putText(pause_view, "-- PAUSED --", (240, 50),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow(window_name, pause_view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): # 'q' atau ESC
                print("[INFO] Keluar...")
                break
            elif key == ord(' '): # Spasi (Pause/Resume)
                paused = not paused
                print(f"[INFO] Stream {'PAUSED' if paused else 'RESUMED'}")
            elif key == ord('s'): # 's' (Snapshot)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = os.path.join(args.save_dir, f"snap_{ts}.jpg")
                if 'frame' in locals() and frame is not None:
                    cv2.imwrite(fn, frame)
                    print(f"[INFO] Frame snapshot tersimpan di: {fn}")

    except KeyboardInterrupt:
        print("\n[INFO] Dibatalkan pengguna.")
    finally:
        ser.close()
        cv2.destroyAllWindows()
        print("[INFO] Koneksi serial ditutup. Selesai.")


if __name__ == "__main__":
    main()

