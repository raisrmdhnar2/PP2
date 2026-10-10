#!/usr/bin/env python3
"""
upload_infer_grove.py — Script CLI Pengiriman Gambar dari Terminal ke
Seeed Grove Vision AI Module V2 (Firmware: tflm_pp2_full_v1)
"""

import os
import glob
import time
import json
import struct
import subprocess
import argparse
import serial
import numpy as np
from PIL import Image

BAUD_RATE = 921600
IMG_SIZE  = (160, 160)


def auto_detect_serial_port(preferred="/dev/ttyACM0"):
    if preferred and os.path.exists(preferred):
        return preferred
    ports = sorted(glob.glob("/dev/ttyACM*"))
    if ports:
        return ports[0]
    return preferred or "/dev/ttyACM0"


import signal
import termios


def is_picocom_active():
    """Hanya mengembalikan True jika picocom/minicom benar-benar aktif (bukan Stopped/T akibat Ctrl+Z)."""
    try:
        out = subprocess.check_output(["pgrep", "-f", "picocom|minicom"], text=True).strip()
        active_found = False
        for line in out.splitlines():
            pid_str = line.strip()
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            try:
                state = subprocess.check_output(["ps", "-o", "state=", "-p", str(pid)], text=True).strip()
                if state.startswith("T"):
                    # Bersihkan proses picocom yang menggantung (Stopped akibat Ctrl+Z)
                    os.kill(pid, signal.SIGKILL)
                elif state:
                    active_found = True
            except Exception:
                pass
        return active_found
    except subprocess.CalledProcessError:
        return False


def send_image_to_grove(image_input, name_label="", port=None, baud=BAUD_RATE):
    port = auto_detect_serial_port(port)
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)
        rgb_arr = np.asarray(pil_img, dtype=np.uint8)
        if not name_label:
            name_label = os.path.basename(image_input)
    else:
        rgb_arr = image_input.astype(np.uint8)
        if not name_label:
            name_label = "Synthetic_160x160"

    raw_bytes = rgb_arr.tobytes()
    checksum  = sum(raw_bytes) & 0xFFFF
    crc_bytes = struct.pack("<H", checksum)
    payload   = raw_bytes + crc_bytes

    picocom_open = is_picocom_active()
    logs = []
    res_json = None
    chunk_size = 256

    if picocom_open:
        # Mode Write-Only Murni: tidak menyentuh termios/DTR/RX buffer milik picocom
        fd = os.open(port, os.O_WRONLY | os.O_NOCTTY)
        try:
            os.write(fd, b"PP2IR")
            termios.tcdrain(fd)
            time.sleep(0.005)  # 5ms cukup bagi MCU untuk membalas RDY (54us) dan masuk loop

            for idx in range(0, len(payload), chunk_size):
                os.write(fd, payload[idx:idx + chunk_size])
                time.sleep(0.003)

            termios.tcdrain(fd)
            time.sleep(0.10)
        finally:
            os.close(fd)

        print(f"[SENT] {len(payload)} byte ({name_label} | Checksum: 0x{checksum:04X}) berhasil dikirim ke {port}!")
        print("[INFO] Karena 'picocom' sedang aktif di terminal lain, hasil inferensi NPU langsung tampil di layar picocom Anda.")
        return None, []

    with serial.Serial(port, baud, timeout=0.2, dsrdtr=False, rtscts=False) as ser:
        # Matikan HUPCL agar saat port ditutup tidak me-reset chip HX6538
        try:
            attrs = termios.tcgetattr(ser.fileno())
            attrs[2] &= ~termios.HUPCL
            termios.tcsetattr(ser.fileno(), termios.TCSANOW, attrs)
        except Exception:
            pass

        # Tunggu melewati jendela 100ms bootloader jika pembukaan port memicu reset
        time.sleep(0.20)
        if ser.in_waiting > 0:
            boot_check = ser.read(ser.in_waiting).decode("utf-8", errors="ignore")
            if "1st BL" in boot_check or "waiting input key" in boot_check:
                time.sleep(0.65)
                if ser.in_waiting > 0:
                    ser.read(ser.in_waiting)

        # 1. Kirim Command Header "PP2IR" dan tunggu balasan "RDY"
        ser.write(b"PP2IR")
        ser.flush()

        t_rdy = time.time()
        while time.time() - t_rdy < 1.0:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if "RDY" in line:
                break

        # 2. Alirkan 76.800 byte piksel RGB + 2 byte Checksum (256 byte per paket)
        for idx in range(0, len(payload), chunk_size):
            ser.write(payload[idx:idx + chunk_size])
            time.sleep(0.003)
        ser.flush()

        print(f"[SENT] {len(payload)} byte ({name_label} | Checksum: 0x{checksum:04X}) berhasil dikirim ke {port}!")

        # 3. Baca seluruh laporan & PP2_JSON langsung di sini
        t0 = time.time()
        while time.time() - t0 < 3.5:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            logs.append(line)
            if line.startswith("PP2_JSON:"):
                try:
                    res_json = json.loads(line[len("PP2_JSON:"):].strip())
                except Exception:
                    pass
            elif "[READY] Menunggu upload gambar berikutnya" in line and res_json is not None:
                break

    return res_json, logs


def main():
    parser = argparse.ArgumentParser(description="Upload Gambar Manual ke Grove Vision AI V2 (tflm_pp2_full_v1)")
    parser.add_argument("--port", default=None, help="Port serial Grove Vision AI V2 (otomatis deteksi /dev/ttyACM0 atau /dev/ttyACM1)")
    parser.add_argument("--baud", type=int, default=BAUD_RATE, help="Baudrate serial (default: 921600)")
    parser.add_argument("--image", default=None, help="Path file gambar tunggal untuk dikirim ke Grove")
    parser.add_argument("--blank", action="store_true", help="Kirim gambar kosong (Blank / No Egg) ke Grove")
    args = parser.parse_args()

    active_port = auto_detect_serial_port(args.port)

    if args.blank:
        blank_rgb = np.full((160, 160, 3), 128, dtype=np.uint8)
        res, logs = send_image_to_grove(blank_rgb, name_label="Blank_Frame", port=active_port, baud=args.baud)
        for l in logs:
            print(l)
        return

    if args.image:
        res, logs = send_image_to_grove(args.image, port=active_port, baud=args.baud)
        for l in logs:
            print(l)
        return

    base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir  = os.path.join(base_dir, "data", "Sexing-Chicken-2", "test")
    hen_list  = sorted(glob.glob(os.path.join(data_dir, "hen", "*.jpg")))
    cock_list = sorted(glob.glob(os.path.join(data_dir, "cock", "*.jpg")))

    blank_rgb = np.full((160, 160, 3), 128, dtype=np.uint8)
    img_h = np.asarray(Image.open(hen_list[0]).convert("RGB").resize(IMG_SIZE), dtype=np.float32)
    img_c = np.asarray(Image.open(cock_list[1]).convert("RGB").resize(IMG_SIZE), dtype=np.float32)
    rng = np.random.default_rng(7)
    ambig_rgb = np.clip(0.32 * img_h + 0.68 * np.flipud(img_c) + rng.normal(0, 18, (160, 160, 3)), 0, 255).astype(np.uint8)

    test_cases = [
        (hen_list[0],  "Betina #1 (hen)",      0),
        (cock_list[0], "Jantan #1 (cock)",     1),
        (blank_rgb,    "Blank (No Egg)",       2),
        (ambig_rgb,    "Ambigu (Low <80%)",    2),
    ]

    for inp, name, expected_cls in test_cases:
        print(f"\n>>> Mengirim sampel: {name} ...")
        res, logs = send_image_to_grove(inp, name_label=name, port=active_port, baud=args.baud)
        for l in logs:
            print(l)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
