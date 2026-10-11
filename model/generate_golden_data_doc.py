#!/usr/bin/env python3
"""
Generator Golden Data untuk Simulasi On-Device PP2 DOC Sexing (Day-Old Chick)
Target: Seeed Grove Vision AI Module V2 (Cortex-M55 + Ethos-U55 NPU)

Menghasilkan file header C/C++ `golden_data_doc.h` berisi 8 sampel uji terkuantisasi INT8:
  - 3 Sampel Class 0 (Female / Betina) -> P(Female) >= 80%, Servo 0°
  - 3 Sampel Class 1 (Male / Jantan)   -> P(Male) >= 80%,   Servo 180°
  - 1 Sampel Blank Frame (std < 12.0)  -> UNDEFINED (Label 2), Servo 90°
  - 1 Sampel Low Confidence (max < 80%)-> UNDEFINED (Label 2), Servo 90°

Semua path menggunakan path relatif terhadap direktori proyek PP2.
"""

import os
import sys
import glob
import pathlib
import argparse
import numpy as np
from PIL import Image

# Path Relatif Portabel
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MODEL_PATH = SCRIPT_DIR / "model_doc_int8.tflite"
DATASET_AUG_DIR = BASE_DIR / "data" / "doc_augmented"
DATASET_RAW_DIR = BASE_DIR / "data" / "doc"
DEFAULT_OUTPUT_HEADER = SCRIPT_DIR / "golden_data_doc.h"

IMG_SIZE = (160, 160)
BLANK_STD_THRESHOLD = 12.0
CONFIDENCE_THRESHOLD = 0.80


def quantize_pixels(img_float, scale, zero_point):
    """Kuantisasi float32 [0..255] -> int8 [-128..127]."""
    q = np.round(img_float / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize(int8_val, scale, zero_point):
    """Dekuantisasi int8 [-128..127] -> float32."""
    return (int8_val.astype(np.float32) - zero_point) * scale


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def load_image_raw_and_int8(path, scale, zero_point):
    with Image.open(path) as img:
        img = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
        raw_px = np.array(img, dtype=np.float32)
    std_val = float(np.std(raw_px))
    int8_data = quantize_pixels(raw_px, scale, zero_point)
    return raw_px, int8_data, std_val


def get_interpreter(model_path):
    """Memuat TFLite interpreter (mencoba tf.lite atau tflite_runtime)."""
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=str(model_path))
        except ImportError:
            raise RuntimeError("Dibutuhkan 'tensorflow' atau 'tflite-runtime' untuk menjalankan inferensi golden data.")
    interpreter.allocate_tensors()
    return interpreter


def main():
    parser = argparse.ArgumentParser(description="Generator Golden Data DOC INT8 untuk Grove Vision AI V2")
    parser.add_argument("--model", type=pathlib.Path, default=MODEL_PATH, help="Path ke model_doc_int8.tflite")
    parser.add_argument("--out-header", type=pathlib.Path, default=DEFAULT_OUTPUT_HEADER, help="Path output golden_data_doc.h")
    args = parser.parse_args()

    print("=" * 70)
    print("  GENERATOR GOLDEN DATA DOC (DAY-OLD CHICK) INT8")
    print("=" * 70)
    print(f"Target Model   : {args.model}")
    print(f"Output Header  : {args.out_header}")

    if not args.model.exists():
        print(f"\n[WARN] File model '{args.model}' belum ditemukan.")
        print("       Silakan jalankan training terlebih dahulu di server (Notebook 4).")
        print("       Skrip ini siap dijalankan kapan saja setelah model diekspor.")
        return

    interpreter = get_interpreter(args.model)
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    in_scale = float(in_det["quantization"][0])
    in_zp = int(in_det["quantization"][1])
    out_scale = float(out_det["quantization"][0])
    out_zp = int(out_det["quantization"][1])

    print(f"[MODEL] Input  INT8: Scale={in_scale:.6f}, ZeroPoint={in_zp}")
    print(f"[MODEL] Output INT8: Scale={out_scale:.6f}, ZeroPoint={out_zp}")

    def run_infer(img_int8):
        inp_batch = np.expand_dims(img_int8, axis=0)
        interpreter.set_tensor(in_det["index"], inp_batch)
        interpreter.invoke()
        raw_out = interpreter.get_tensor(out_det["index"])[0]
        f_logits = dequantize(raw_out, out_scale, out_zp)
        probs = softmax(f_logits)
        return raw_out, probs

    # Cari citra uji dari doc_augmented (test folder) atau fallback ke doc
    candidate_female = []
    candidate_male = []

    search_dirs = [
        DATASET_AUG_DIR / "test",
        DATASET_AUG_DIR / "valid",
        DATASET_RAW_DIR
    ]

    for d in search_dirs:
        if (d / "Female").exists():
            candidate_female.extend(sorted((d / "Female").glob("*.jpg")))
        if (d / "Male").exists():
            candidate_male.extend(sorted((d / "Male").glob("*.jpg")))

    print(f"[DATA] Kandidat citra: {len(candidate_female)} Female, {len(candidate_male)} Male.")

    selected_females = []
    selected_males = []
    low_conf_candidates = []

    for fpath in candidate_female:
        raw_px, img_int8, px_std = load_image_raw_and_int8(fpath, in_scale, in_zp)
        raw_out, probs = run_infer(img_int8)
        conf = probs[0]
        if conf >= CONFIDENCE_THRESHOLD and px_std >= BLANK_STD_THRESHOLD:
            selected_females.append({
                "path": fpath, "raw_px": raw_px, "img_int8": img_int8,
                "raw_out": raw_out, "probs": probs, "px_std": px_std, "conf": conf
            })
        elif conf < CONFIDENCE_THRESHOLD and px_std >= BLANK_STD_THRESHOLD:
            low_conf_candidates.append({
                "path": fpath, "raw_px": raw_px, "img_int8": img_int8,
                "raw_out": raw_out, "probs": probs, "px_std": px_std, "conf": max(probs)
            })

    for mpath in candidate_male:
        raw_px, img_int8, px_std = load_image_raw_and_int8(mpath, in_scale, in_zp)
        raw_out, probs = run_infer(img_int8)
        conf = probs[1]
        if conf >= CONFIDENCE_THRESHOLD and px_std >= BLANK_STD_THRESHOLD:
            selected_males.append({
                "path": mpath, "raw_px": raw_px, "img_int8": img_int8,
                "raw_out": raw_out, "probs": probs, "px_std": px_std, "conf": conf
            })
        elif conf < CONFIDENCE_THRESHOLD and px_std >= BLANK_STD_THRESHOLD:
            low_conf_candidates.append({
                "path": mpath, "raw_px": raw_px, "img_int8": img_int8,
                "raw_out": raw_out, "probs": probs, "px_std": px_std, "conf": max(probs)
            })

    selected_females.sort(key=lambda x: -x["conf"])
    selected_males.sort(key=lambda x: -x["conf"])

    samples = []
    # 3 Sampel Female
    for i in range(min(3, len(selected_females))):
        s = selected_females[i]
        samples.append({
            "name": f"DOC_Female_{i+1:02d}",
            "img_int8": s["img_int8"],
            "raw_c0": int(s["raw_out"][0]),
            "raw_c1": int(s["raw_out"][1]),
            "expected_label": 0,
            "std_x10": int(round(s["px_std"] * 10)),
            "probs": s["probs"],
            "src": s["path"].name
        })

    # 3 Sampel Male
    for i in range(min(3, len(selected_males))):
        s = selected_males[i]
        samples.append({
            "name": f"DOC_Male_{i+1:02d}",
            "img_int8": s["img_int8"],
            "raw_c0": int(s["raw_out"][0]),
            "raw_c1": int(s["raw_out"][1]),
            "expected_label": 1,
            "std_x10": int(round(s["px_std"] * 10)),
            "probs": s["probs"],
            "src": s["path"].name
        })

    # 1 Sampel Blank Frame (Piksel seragam, std = 0 < 12.0)
    blank_px = np.full((160, 160, 3), 235.0, dtype=np.float32)
    blank_std = float(np.std(blank_px))
    blank_int8 = quantize_pixels(blank_px, in_scale, in_zp)
    blank_out, blank_probs = run_infer(blank_int8)
    samples.append({
        "name": "Blank_Frame",
        "img_int8": blank_int8,
        "raw_c0": int(blank_out[0]),
        "raw_c1": int(blank_out[1]),
        "expected_label": 2, # UNDEFINED karena std < 12.0
        "std_x10": int(round(blank_std * 10)),
        "probs": blank_probs,
        "src": "synthetic_blank_235"
    })

    # 1 Sampel Low Confidence / Ambiguous
    if low_conf_candidates:
        lc = min(low_conf_candidates, key=lambda x: abs(x["conf"] - 0.60))
        lc_src = lc["path"].name
    elif selected_females and selected_males:
        blend_px = 0.5 * selected_females[0]["raw_px"] + 0.5 * selected_males[0]["raw_px"]
        blend_std = float(np.std(blend_px))
        blend_int8 = quantize_pixels(blend_px, in_scale, in_zp)
        blend_out, blend_probs = run_infer(blend_int8)
        lc = {
            "px_std": blend_std, "img_int8": blend_int8,
            "raw_out": blend_out, "probs": blend_probs
        }
        lc_src = "synthetic_blend_50_50"
    else:
        lc_src = "fallback_flat"
        lc = {
            "px_std": 15.0, "img_int8": np.zeros((160, 160, 3), dtype=np.int8),
            "raw_out": [0, 0], "probs": [0.5, 0.5]
        }

    samples.append({
        "name": "Ambigu_LowConf",
        "img_int8": lc["img_int8"],
        "raw_c0": int(lc["raw_out"][0]),
        "raw_c1": int(lc["raw_out"][1]),
        "expected_label": 2, # UNDEFINED karena confidence < 80%
        "std_x10": int(round(lc["px_std"] * 10)),
        "probs": lc["probs"],
        "src": lc_src
    })

    # Tulis file header C/C++
    args.out_header.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_header, "w", encoding="utf-8") as f:
        f.write("// Auto-generated Golden Data untuk Simulasi AI PP2 DOC Sexing (Day-Old Chick)\n")
        f.write("// Model: MobileNetV2 (alpha=0.35) INT8 [1, 160, 160, 3]\n")
        f.write("#ifndef GOLDEN_DATA_DOC_H\n")
        f.write("#define GOLDEN_DATA_DOC_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("#define NUM_GOLDEN_SAMPLES 8\n")
        f.write("#define GOLDEN_INPUT_SIZE  76800  // 160 * 160 * 3 bytes\n\n")

        f.write("static const char* golden_sample_names[NUM_GOLDEN_SAMPLES] = {\n")
        for s in samples:
            f.write(f'    "{s["name"]}",\n')
        f.write("};\n\n")

        f.write("// Format: {ref_raw_c0, ref_raw_c1, expected_label (0=Female, 1=Male, 2=Undefined), pixel_std_x10}\n")
        f.write("static const int16_t golden_output[NUM_GOLDEN_SAMPLES][4] = {\n")
        for s in samples:
            f.write(f"    {{{s['raw_c0']:4d}, {s['raw_c1']:4d}, {s['expected_label']}, {s['std_x10']:3d}}}, // {s['name']} ({s['src']})\n")
        f.write("};\n\n")

        f.write('__attribute__((aligned(16), section(".algo"))) static const int8_t golden_input[NUM_GOLDEN_SAMPLES * GOLDEN_INPUT_SIZE] = {\n')
        all_bytes = np.concatenate([s["img_int8"].flatten() for s in samples])
        chunk_size = 32
        lines = []
        for idx in range(0, len(all_bytes), chunk_size):
            chunk = all_bytes[idx : idx + chunk_size]
            lines.append("    " + ",".join(str(int(v)) for v in chunk) + ",\n")
        f.writelines(lines)
        f.write("};\n\n")
        f.write("#endif // GOLDEN_DATA_DOC_H\n")

    size_mb = os.path.getsize(args.out_header) / (1024 * 1024)
    print(f"\n[SUCCESS] Header C/C++ berhasil dibuat: {args.out_header} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()

