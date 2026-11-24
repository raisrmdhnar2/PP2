#!/usr/bin/env python3
"""
grove_serial.py — Modul Komunikasi Serial High-Speed antara PC dan
Seeed Grove Vision AI Module V2 (Firmware: tflm_pp2_full_v1)
Mendukung Protokol 'PP2IP' (Ping) dan 'PP2IR' (Raw 160x160 RGB Image Upload).
"""

import os
import glob
import time
import json
import struct
import io
import serial
import serial.tools.list_ports
import numpy as np
from PIL import Image, ImageOps

import threading

BAUD_RATE = 921600
IMG_SIZE  = (160, 160)
IMG_BYTES = 160 * 160 * 3  # 76800 bytes


def list_available_ports():
    """Mendeteksi daftar port serial yang tersedia di sistem (hanya USB/ACM/Grove)."""
    found = []
    # 1. Menggunakan pyserial list_ports
    for p in serial.tools.list_ports.comports():
        dev = p.device
        desc = p.description or ""
        hwid = p.hwid or ""
        # Abaikan port onboard /dev/ttyS*
        if dev.startswith("/dev/ttyS"):
            continue
        is_grove = any(k in (desc + hwid + dev).lower() for k in ["acm", "grove", "seeed", "vision", "usb"])
        found.append({
            "device": dev,
            "description": desc,
            "is_recommended": is_grove
        })
    # 2. Fallback scan /dev/ttyACM* & /dev/ttyUSB*
    dev_nodes = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    for node in dev_nodes:
        if not any(item["device"] == node for item in found):
            found.append({
                "device": node,
                "description": "USB Serial Device",
                "is_recommended": True
            })
    return sorted(found, key=lambda x: (not x["is_recommended"], x["device"]))


def get_default_port(preferred="/dev/ttyACM0"):
    """Mengembalikan port serial default / terbaik."""
    ports = list_available_ports()
    for p in ports:
        if p["device"] == preferred:
            return preferred
    for p in ports:
        if p["is_recommended"]:
            return p["device"]
    if ports:
        return ports[0]["device"]
    return preferred or "/dev/ttyACM0"


def preprocess_image_to_rgb160(image_input):
    """
    Mengubah berbagai bentuk input (PIL Image, NumPy array, file path, atau bytes)
    menjadi (pil_160, rgb_array_160, raw_bytes, checksum).

    Langkah Preprocessing Standar Grove NPU:
    1. Koreksi otomatis orientasi EXIF (misal foto smartphone yang terbalik/miring).
    2. Penanganan alpha channel (RGBA/LA) dengan komposisi latar belakang putih/netral.
    3. Center-Crop presisi 1:1 aspek rasio untuk menghindari distorsi/gepeng pada bulu sayap DOC.
    4. Resampling antialiasing berkualitas tinggi (Lanczos Filter) ke resolusi target 160x160 RGB.
    5. Ekstraksi raw byte buffer (76.800 B) dan kalkulasi 16-bit Little-Endian checksum.
    """
    if isinstance(image_input, (str, bytes, io.BytesIO)):
        if isinstance(image_input, str) and os.path.exists(image_input):
            pil_raw = Image.open(image_input)
        elif isinstance(image_input, bytes):
            pil_raw = Image.open(io.BytesIO(image_input))
        else:
            pil_raw = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3 and image_input.shape[2] == 3:
            pil_raw = Image.fromarray(image_input.astype(np.uint8), mode="RGB")
        elif image_input.ndim == 2:
            pil_raw = Image.fromarray(image_input.astype(np.uint8), mode="L")
        else:
            raise ValueError(f"Format NumPy array tidak didukung: {image_input.shape}")
    elif isinstance(image_input, Image.Image):
        pil_raw = image_input
    else:
        raise TypeError(f"Tipe input citra tidak dikenali: {type(image_input)}")

    # 1. Koreksi otomatis orientasi EXIF
    try:
        pil_raw = ImageOps.exif_transpose(pil_raw)
    except Exception:
        pass

    # 2. Penanganan kanal alpha (transparansi) secara aman
    if pil_raw.mode in ("RGBA", "LA") or (pil_raw.mode == "P" and "transparency" in pil_raw.info):
        rgba = pil_raw.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        pil_img = bg
    else:
        pil_img = pil_raw.convert("RGB")

    # 3. Center-Crop menjadi bujur sangkar (1:1 aspect ratio)
    w, h = pil_img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    square_img = pil_img.crop((left, top, right, bottom))

    # 4. Resize ke 160x160 menggunakan filter Lanczos beresolusi tinggi (melindungi detail helai bulu sayap)
    resample_filter = getattr(Image.Resampling, "LANCZOS", Image.Resampling.BICUBIC)
    resized_img = square_img.resize(IMG_SIZE, resample_filter)
    rgb_arr = np.asarray(resized_img, dtype=np.uint8)

    # 5. Raw bytes dan checksum 16-bit
    raw_bytes = rgb_arr.tobytes()
    checksum = sum(raw_bytes) & 0xFFFF
    return resized_img, rgb_arr, raw_bytes, checksum


def compute_image_quality(rgb_arr):
    """Menghitung metrik kualitas citra untuk validasi sebelum kirim ke hardware."""
    mean_val = float(np.mean(rgb_arr))
    std_val = float(np.std(rgb_arr))
    is_blank = std_val < 12.0
    is_too_dark = mean_val < 25.0
    is_too_bright = mean_val > 235.0

    if is_blank:
        status = "BLANK_FRAME"
        msg = "Peringatan: Frame terdeteksi kosong/polos (tidak ada objek)"
    elif is_too_dark:
        status = "TOO_DARK"
        msg = "Peringatan: Pencahayaan sangat redup (< 25)"
    elif is_too_bright:
        status = "TOO_BRIGHT"
        msg = "Peringatan: Pencahayaan terlalu silau (> 235)"
    else:
        status = "OPTIMAL"
        msg = "Kualitas citra optimal untuk inferensi NPU"

    return {
        "mean_brightness": round(mean_val, 1),
        "contrast_std": round(std_val, 1),
        "status": status,
        "message": msg
    }


class GroveVisionClient:
    """Klien serial presisten untuk mengirim frame dan menerima hasil inferensi NPU dari Grove Vision AI V2."""

    def __init__(self, port=None, baud=BAUD_RATE):
        self.port = port or get_default_port()
        self.baud = baud
        self.ser = None
        self.lock = threading.Lock()

    def _ensure_connected(self, port=None):
        """Memastikan koneksi serial terbuka dan board dalam keadaan siap."""
        active_port = port or self.port
        if self.ser is not None and self.ser.is_open:
            if self.ser.port == active_port:
                return self.ser
            else:
                self.close()

        if not os.path.exists(active_port):
            raise FileNotFoundError(f"Port {active_port} tidak ditemukan di sistem")

        self.port = active_port
        ser = serial.Serial(active_port, self.baud, timeout=0.5, dsrdtr=False, rtscts=False)
        
        # Beri waktu board Himax WE2 menyelesaikan bootloader (~1.2 - 1.5 detik)
        t0 = time.time()
        while time.time() - t0 < 1.6:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore")
                if "[READY]" in line or "PP2_JSON" in line:
                    break
            else:
                time.sleep(0.05)

        ser.reset_input_buffer()
        ser.reset_output_buffer()
        self.ser = ser
        return self.ser

    def close(self):
        """Menutup koneksi serial jika sedang terbuka."""
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def ping(self, port=None):
        """Kirim command PING 'PP2IP' ke modul Grove untuk memastikan respon."""
        active_port = port or self.port
        with self.lock:
            try:
                ser = self._ensure_connected(active_port)
                ser.reset_input_buffer()
                ser.write(b"PP2IP\n")
                ser.flush()

                t0 = time.time()
                while time.time() - t0 < 1.0:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if "PP2_PONG:READY" in line or "PP2_JSON" in line:
                        return {"success": True, "port": active_port, "status": "ONLINE"}
                    if "RDY" in line:
                        return {"success": True, "port": active_port, "status": "READY_FOR_IMAGE"}
                    if line:
                        time.sleep(0.01)

                return {"success": True, "port": active_port, "status": "CONNECTED_NO_PONG"}
            except Exception as e:
                self.close()
                return {"success": False, "port": active_port, "error": str(e)}

    def infer(self, image_input, port=None):
        """
        Kirim frame citra (160x160 RGB uint8 + CRC16) ke Grove via protokol 'PP2IR'.
        Mengembalikan respon JSON inferensi NPU dan log serial.
        """
        active_port = port or self.port
        pil_160, rgb_arr, raw_bytes, checksum = preprocess_image_to_rgb160(image_input)
        crc_bytes = struct.pack("<H", checksum)
        payload = raw_bytes + crc_bytes
        chunk_size = 256

        logs = []
        parsed_json = None
        t_start_total = time.time()

        with self.lock:
            try:
                ser = self._ensure_connected(active_port)

                # 1. Bersihkan sisa buffer
                ser.reset_input_buffer()

                # 2. Kirim Header Handshake 'PP2IR'
                ser.write(b"PP2IR")
                ser.flush()

                # Tunggu sinyal 'RDY' dari Grove MCU
                t_rdy = time.time()
                rdy_received = False
                while time.time() - t_rdy < 1.2:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if "RDY" in line:
                        rdy_received = True
                        break
                    elif line:
                        logs.append(line)

                if not rdy_received:
                    return {
                        "success": False,
                        "error": "Timeout menunggu 'RDY' dari Grove Vision AI",
                        "logs": logs,
                        "rdy_ok": False
                    }

                # 3. Alirkan Payload 76.802 Byte persis seperti upload_infer_grove.py
                t_send_start = time.time()
                for idx in range(0, len(payload), chunk_size):
                    ser.write(payload[idx:idx + chunk_size])
                    time.sleep(0.003)
                ser.flush()
                send_duration_ms = (time.time() - t_send_start) * 1000

                # 4. Dengarkan Respon Hasil Inferensi NPU
                t_listen = time.time()
                while time.time() - t_listen < 4.0:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    logs.append(line)
                    if line.startswith("PP2_JSON:"):
                        try:
                            json_str = line[len("PP2_JSON:"):].strip()
                            parsed_json = json.loads(json_str)
                            break
                        except Exception:
                            pass
                    elif "PP2_ERR:" in line:
                        logs.append(line)
                        break

                total_latency_ms = (time.time() - t_start_total) * 1000

                # 5. Format hasil visual
                if parsed_json is not None:
                    class_id = parsed_json.get("class_id", 2)
                    conf_pct = float(parsed_json.get("conf", parsed_json.get("confidence", 0.0)))
                    infer_ms = parsed_json.get("npu_ms", parsed_json.get("infer_ms", 0))
                    i2c_stat = parsed_json.get("i2c_status", "UNKNOWN")
                    raw_label = parsed_json.get("label", parsed_json.get("class_name", ""))

                    if class_id == 0:
                        label_display = "BETINA (FEMALE)"
                        color = "#10B981"  # Emerald Green
                        badge_type = "female"
                    elif class_id == 1:
                        label_display = "JANTAN (MALE)"
                        color = "#3B82F6"  # Blue
                        badge_type = "male"
                    else:
                        label_display = "BLANK / AMBIGU"
                        color = "#F59E0B"  # Amber
                        badge_type = "ambiguous"

                    return {
                        "success": True,
                        "class_id": class_id,
                        "class_name": raw_label or label_display,
                        "label_display": label_display,
                        "badge_type": badge_type,
                        "color": color,
                        "confidence": conf_pct,
                        "infer_ms": infer_ms,
                        "i2c_status": i2c_stat,
                        "total_ms": round(total_latency_ms, 1),
                        "send_ms": round(send_duration_ms, 1),
                        "checksum": f"0x{checksum:04X}",
                        "raw_json": parsed_json,
                        "logs": logs[-10:] if len(logs) > 10 else logs
                    }
                else:
                    return {
                        "success": False,
                        "error": "Tidak menerima respon 'PP2_JSON:' dari modul Grove",
                        "logs": logs,
                        "rdy_ok": rdy_received
                    }

            except Exception as e:
                self.close()
                return {
                    "success": False,
                    "error": f"Kesalahan Serial I/O: {str(e)}",
                    "logs": logs
                }

