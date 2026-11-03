#!/usr/bin/env python3
"""
One-Click Flashing Tool for Seeed Grove Vision AI V2
Flashes both the Firmware (output.img) and the Vela NPU Model (0x3AB7B000).

Usage:
    conda activate tinyml
    python PP2/firmware/flash_firmware.py [--port /dev/ttyACM0] [--model doc]
"""

import os
import sys
import argparse
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
XMODEM_SCRIPT = os.path.join(PROJECT_ROOT, "tinyml-arm-workflow/third_party/Seeed_Grove_Vision_AI_Module_V2/xmodem/xmodem_send.py")
FIRMWARE_IMG = os.path.join(PROJECT_ROOT, "PP2/firmware/tflm_pp2_full_cam/output.img")
MODEL_DOC    = os.path.join(PROJECT_ROOT, "PP2/model/vela_out/model_doc_int8_vela.tflite")
MODEL_BASE   = os.path.join(PROJECT_ROOT, "PP2/model/vela_out/model_int8_vela.tflite")

FLASH_ADDR   = "0x3AB7B000"
FLASH_OFFSET = "0x00B7B000"

def parse_args():
    parser = argparse.ArgumentParser(description="Flash Firmware & Vela Model to Grove Vision AI V2")
    parser.add_argument("--port", "-p", default="/dev/ttyACM0", help="Serial port (default: /dev/ttyACM0)")
    parser.add_argument("--baud", "-b", type=int, default=921600, help="Baudrate (default: 921600)")
    parser.add_argument("--model", "-m", choices=["doc", "base"], default="doc",
                        help="Model yang digunakan: 'doc' (model_doc_int8_vela.tflite) atau 'base' (model_int8_vela.tflite)")
    parser.add_argument("--firmware-only", action="store_true", help="Hanya flash firmware tanpa model")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(FIRMWARE_IMG):
        print(f"[ERROR] File firmware tidak ditemukan di: {FIRMWARE_IMG}")
        sys.exit(1)

    selected_model = MODEL_DOC if args.model == "doc" else MODEL_BASE
    if not os.path.exists(selected_model) and not args.firmware_only:
        print(f"[ERROR] File model vela tidak ditemukan di: {selected_model}")
        sys.exit(1)

    print("===============================================================")
    print(" Grove Vision AI V2 - Firmware & Model Flasher (XMODEM)")
    print("===============================================================")
    print(f" Port      : {args.port}")
    print(f" Baudrate  : {args.baud}")
    print(f" Firmware  : {FIRMWARE_IMG}")
    if not args.firmware_only:
        print(f" Model     : {selected_model}")
        print(f" Target Adr: {FLASH_ADDR} (offset {FLASH_OFFSET})")
    print("===============================================================")
    print("\n[PETUNJUK PENTING]")
    print(" 1. Sebelum menekan Enter, pastikan Grove Vision AI V2 masuk ke bootloader mode:")
    print("    - Tekan tombol BOOT (tahan), tekan RESET sekali, lalu lepas BOOT.")
    print("    - Atau cabut dan colok kembali kabel USB.")
    print(" 2. Pastikan port serial tidak sedang dibuka oleh Serial Monitor / Python Viewer.")
    print("")

    input("Tekan [ENTER] untuk memulai proses flashing...")

    cmd = [
        sys.executable,
        XMODEM_SCRIPT,
        f"--port={args.port}",
        f"--baudrate={args.baud}",
        f"--file={FIRMWARE_IMG}"
    ]

    if not args.firmware_only:
        cmd.append(f'--model="{selected_model} {FLASH_ADDR} {FLASH_OFFSET}"')

    cmd_str = " ".join(cmd)
    print(f"\n[EXEC] {cmd_str}\n")
    ret = subprocess.call(cmd_str, shell=True)

    if ret == 0:
        print("\n[SUKSES] Flashing firmware dan model berhasil!")
        print("Modul Grove Vision AI V2 akan restart otomatis.")
        print(f"Jalankan viewer dengan: python PP2/firmware/cam_viewer.py --port {args.port}\n")
    else:
        print(f"\n[GAGAL] Flashing gagal dengan kode error {ret}. Periksa koneksi USB dan coba lagi.")

if __name__ == "__main__":
    main()

