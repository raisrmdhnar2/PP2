#!/usr/bin/env python3
"""
Generator Golden Data untuk Simulasi On-Device PP2 (Deteksi Jenis Kelamin Telur Ayam)
Target: Seeed Grove Vision AI Module V2 (Cortex-M55 + Ethos-U55 NPU)

Menghasilkan file header C/C++ `golden_data.h` berisi 8 sampel uji terkuantisasi INT8 (600 KB):
  - 3 Sampel Class 0 (Betina / Hen)  -> P(Betina) >= 80%, Servo 0°
  - 3 Sampel Class 1 (Jantan / Cock) -> P(Jantan) >= 80%, Servo 180°
  - 1 Sampel Blank Frame (Terang merata, std < 12.0) -> UNDEFINED (Label 2), Servo 90°
  - 1 Sampel Low Confidence (max(probs) < 80%)       -> UNDEFINED (Label 2), Servo 90°
"""

import glob
import os
import sys
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_int8.tflite")
DATASET_DIR = os.path.join(BASE_DIR, "data", "Sexing-Chicken-2")
OUTPUT_HEADER = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "tinyml-arm-workflow",
        "third_party",
        "Seeed_Grove_Vision_AI_Module_V2",
        "EPII_CM55M_APP_S",
        "app",
        "scenario_app",
        "tflm_pp2_simulasi",
        "golden_data.h",
    )
)

VELA_SITE_PACKAGES = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "tinyml-arm-workflow",
        "third_party",
        "ml-embedded-evaluation-kit",
        "resources_downloaded",
        "env",
        "lib",
        "python3.12",
        "site-packages",
    )
)

IMG_SIZE = (160, 160)
BLANK_STD_THRESHOLD = 12.0
CONFIDENCE_THRESHOLD = 0.80


class VelaTFLiteInt8Evaluator:
    """Evaluator model_int8.tflite berbasis ethosu.vela + NumPy (tanpa ketergantungan TensorFlow)."""

    def __init__(self, model_path):
        if VELA_SITE_PACKAGES not in sys.path and os.path.isdir(VELA_SITE_PACKAGES):
            sys.path.append(VELA_SITE_PACKAGES)
        from ethosu.vela import tflite_reader
        from ethosu.vela.operation import Padding

        self.Padding = Padding
        nng = tflite_reader.read_tflite(
            model_path,
            batch_size=1,
            feed_dict={},
            output_node_names=[],
            initialisation_nodes=[],
        )
        sg = nng.get_root_subgraph()
        self.ops = sg.get_all_ops()
        in_t = self.ops[0].outputs[0]
        out_t = self.ops[-1].outputs[0]
        self.in_shape = in_t.shape
        self.in_scale = float(in_t.quantization.scale_f32)
        self.in_zp = int(in_t.quantization.zero_point)
        self.out_shape = out_t.shape
        self.out_scale = float(out_t.quantization.scale_f32)
        self.out_zp = int(out_t.quantization.zero_point)

    @staticmethod
    def _get_same_padding(in_size, stride, filter_size):
        out_size = (in_size + stride - 1) // stride
        pad_needed = max(0, (out_size - 1) * stride + filter_size - in_size)
        pad_before = pad_needed // 2
        pad_after = pad_needed - pad_before
        return pad_before, pad_after, out_size

    def invoke(self, img_int8):
        input_batch = np.expand_dims(img_int8, axis=0).astype(np.int32)
        tensors = {}
        for op in self.ops:
            tname = op.type.name
            if tname == "Placeholder":
                tensors[op.outputs[0]] = input_batch
            elif tname == "Const":
                tensors[op.outputs[0]] = op.outputs[0].values
            elif tname == "Pad":
                x = tensors[op.inputs[0]]
                paddings = tensors[op.inputs[1]]
                in_zp = int(op.inputs[0].quantization.zero_point)
                tensors[op.outputs[0]] = np.pad(
                    x, paddings, mode="constant", constant_values=in_zp
                )
            elif tname == "Conv2DBias":
                x_t, w_t, b_t = op.inputs[0], op.inputs[1], op.inputs[2]
                out_t = op.outputs[0]
                x = tensors[x_t].astype(np.int32)
                w = tensors[w_t].astype(np.int32)
                b = tensors[b_t].astype(np.int64)

                in_zp = int(x_t.quantization.zero_point)
                in_scale = float(x_t.quantization.scale_f32)
                w_scales = np.array(w_t.quantization.scale_f32, dtype=np.float64)
                w_zps = np.array(w_t.quantization.zero_point, dtype=np.int32)
                out_scale = float(out_t.quantization.scale_f32)
                out_zp = int(out_t.quantization.zero_point)

                sh, sw = op.attrs["stride_h"], op.attrs["stride_w"]
                fh, fw, _, cout = w.shape
                _, ih, iw, _ = x.shape

                if op.attrs["padding"] == self.Padding.SAME:
                    pt, pb, oh = self._get_same_padding(ih, sh, fh)
                    pl, pr, ow = self._get_same_padding(iw, sw, fw)
                    x_pad = np.pad(
                        x,
                        ((0, 0), (pt, pb), (pl, pr), (0, 0)),
                        mode="constant",
                        constant_values=in_zp,
                    )
                else:
                    oh = (ih - fh) // sh + 1
                    ow = (iw - fw) // sw + 1
                    x_pad = x

                x_shifted = x_pad - in_zp
                w_shifted = w - w_zps.reshape(1, 1, 1, -1)

                acc = np.zeros((1, oh, ow, cout), dtype=np.int64)
                for ky in range(fh):
                    for kx in range(fw):
                        patch = x_shifted[:, ky : ky + oh * sh : sh, kx : kx + ow * sw : sw, :]
                        acc += np.tensordot(patch, w_shifted[ky, kx], axes=([3], [0]))
                acc += b.reshape(1, 1, 1, -1)

                eff_scale = (in_scale * w_scales) / out_scale
                res = np.round(acc * eff_scale.reshape(1, 1, 1, -1)) + out_zp
                if op.activation and op.activation.op_type.name == "Relu6":
                    q_min = max(-128, int(np.round(0.0 / out_scale + out_zp)))
                    q_max = min(127, int(np.round(6.0 / out_scale + out_zp)))
                    res = np.clip(res, q_min, q_max)
                else:
                    res = np.clip(res, -128, 127)
                tensors[out_t] = res.astype(np.int32)

            elif tname == "DepthwiseConv2DBias":
                x_t, w_t, b_t = op.inputs[0], op.inputs[1], op.inputs[2]
                out_t = op.outputs[0]
                x = tensors[x_t].astype(np.int32)
                w = tensors[w_t].astype(np.int32)
                b = tensors[b_t].astype(np.int64)

                in_zp = int(x_t.quantization.zero_point)
                in_scale = float(x_t.quantization.scale_f32)
                w_scales = np.array(w_t.quantization.scale_f32, dtype=np.float64)
                w_zps = np.array(w_t.quantization.zero_point, dtype=np.int32)
                out_scale = float(out_t.quantization.scale_f32)
                out_zp = int(out_t.quantization.zero_point)

                sh, sw = op.attrs["stride_h"], op.attrs["stride_w"]
                fh, fw = w.shape[0], w.shape[1]
                _, ih, iw, cin = x.shape
                w_2d = w.reshape(fh, fw, cin)

                if op.attrs["padding"] == self.Padding.SAME:
                    pt, pb, oh = self._get_same_padding(ih, sh, fh)
                    pl, pr, ow = self._get_same_padding(iw, sw, fw)
                    x_pad = np.pad(
                        x,
                        ((0, 0), (pt, pb), (pl, pr), (0, 0)),
                        mode="constant",
                        constant_values=in_zp,
                    )
                else:
                    oh = (ih - fh) // sh + 1
                    ow = (iw - fw) // sw + 1
                    x_pad = x

                x_shifted = x_pad - in_zp
                w_shifted = w_2d - w_zps.reshape(1, 1, -1)

                acc = np.zeros((1, oh, ow, cin), dtype=np.int64)
                for ky in range(fh):
                    for kx in range(fw):
                        patch = x_shifted[:, ky : ky + oh * sh : sh, kx : kx + ow * sw : sw, :]
                        acc += patch * w_shifted[ky, kx].reshape(1, 1, 1, -1)
                acc += b.reshape(1, 1, 1, -1)

                eff_scale = (in_scale * w_scales) / out_scale
                res = np.round(acc * eff_scale.reshape(1, 1, 1, -1)) + out_zp
                if op.activation and op.activation.op_type.name == "Relu6":
                    q_min = max(-128, int(np.round(0.0 / out_scale + out_zp)))
                    q_max = min(127, int(np.round(6.0 / out_scale + out_zp)))
                    res = np.clip(res, q_min, q_max)
                else:
                    res = np.clip(res, -128, 127)
                tensors[out_t] = res.astype(np.int32)

            elif tname == "Add":
                x1_t, x2_t = op.inputs[0], op.inputs[1]
                out_t = op.outputs[0]
                x1 = (tensors[x1_t].astype(np.float64) - int(x1_t.quantization.zero_point)) * float(
                    x1_t.quantization.scale_f32
                )
                x2 = (tensors[x2_t].astype(np.float64) - int(x2_t.quantization.zero_point)) * float(
                    x2_t.quantization.scale_f32
                )
                out_scale = float(out_t.quantization.scale_f32)
                out_zp = int(out_t.quantization.zero_point)
                res = np.round((x1 + x2) / out_scale) + out_zp
                tensors[out_t] = np.clip(res, -128, 127).astype(np.int32)

            elif tname == "Mean":
                x_t = op.inputs[0]
                out_t = op.outputs[0]
                x = tensors[x_t].astype(np.float64)
                in_zp = int(x_t.quantization.zero_point)
                in_scale = float(x_t.quantization.scale_f32)
                out_scale = float(out_t.quantization.scale_f32)
                out_zp = int(out_t.quantization.zero_point)
                mean_val = np.mean((x - in_zp) * in_scale, axis=(1, 2))
                res = np.round(mean_val / out_scale) + out_zp
                tensors[out_t] = np.clip(res, -128, 127).astype(np.int32)

            elif tname == "FullyConnected":
                x_t, w_t, b_t = op.inputs[0], op.inputs[1], op.inputs[2]
                out_t = op.outputs[0]
                x = tensors[x_t].astype(np.int32) - int(x_t.quantization.zero_point)
                w = tensors[w_t].astype(np.int32) - int(w_t.quantization.zero_point)
                b = tensors[b_t].astype(np.int64)
                if w.shape[0] == x.shape[-1]:
                    acc = np.dot(x, w) + b
                else:
                    acc = np.dot(x, w.T) + b
                eff_scale = (
                    float(x_t.quantization.scale_f32)
                    * float(w_t.quantization.scale_f32)
                    / float(out_t.quantization.scale_f32)
                )
                res = np.round(acc * eff_scale) + int(out_t.quantization.zero_point)
                tensors[out_t] = np.clip(res, -128, 127).astype(np.int32)

            elif tname == "Softmax":
                x_t = op.inputs[0]
                out_t = op.outputs[0]
                x = (tensors[x_t].astype(np.float64) - int(x_t.quantization.zero_point)) * float(
                    x_t.quantization.scale_f32
                )
                e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
                res = np.round(probs / float(out_t.quantization.scale_f32)) + int(
                    out_t.quantization.zero_point
                )
                tensors[out_t] = np.clip(res, -128, 127).astype(np.int8)

        return tensors[self.ops[-1].outputs[0]][0]


def load_and_preprocess_rgb(img_path):
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    raw_pixels = np.array(img_resized, dtype=np.float32)
    return raw_pixels


def quantize_pixels(raw_pixels, in_scale, in_zp):
    img_float = (raw_pixels / 127.5) - 1.0
    img_int8 = np.clip(np.round(img_float / in_scale + in_zp), -128, 127).astype(np.int8)
    return img_int8


def main():
    print(f"[INFO] Memuat model TFLite: {MODEL_PATH}")
    evaluator = VelaTFLiteInt8Evaluator(MODEL_PATH)
    in_scale, in_zp = evaluator.in_scale, evaluator.in_zp
    out_scale, out_zp = evaluator.out_scale, evaluator.out_zp

    print(f"[INFO] Input  : shape={evaluator.in_shape}, scale={in_scale}, zp={in_zp}")
    print(f"[INFO] Output : shape={evaluator.out_shape}, scale={out_scale}, zp={out_zp}")

    def run_infer(img_int8):
        raw_out = evaluator.invoke(img_int8)
        probs = (raw_out.astype(np.float32) - out_zp) * out_scale
        probs = np.clip(probs, 0.0, 1.0)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        return raw_out, probs

    hen_files = sorted(
        glob.glob(os.path.join(DATASET_DIR, "test", "hen", "*.jpg"))
        + glob.glob(os.path.join(DATASET_DIR, "valid", "hen", "*.jpg"))
    )
    cock_files = sorted(
        glob.glob(os.path.join(DATASET_DIR, "test", "cock", "*.jpg"))
        + glob.glob(os.path.join(DATASET_DIR, "valid", "cock", "*.jpg"))
    )

    hen_candidates = []
    cock_candidates = []
    low_conf_candidates = []

    for fpath in hen_files:
        raw_px = load_and_preprocess_rgb(fpath)
        px_std = float(np.std(raw_px))
        img_int8 = quantize_pixels(raw_px, in_scale, in_zp)
        raw_out, probs = run_infer(img_int8)
        conf = float(np.max(probs))
        pred_cls = int(np.argmax(probs))
        entry = {
            "path": fpath,
            "raw_px": raw_px,
            "px_std": px_std,
            "img_int8": img_int8,
            "raw_out": raw_out,
            "probs": probs,
            "conf": conf,
            "pred_cls": pred_cls,
        }
        if px_std >= BLANK_STD_THRESHOLD and pred_cls == 0 and conf >= 0.85:
            hen_candidates.append(entry)
        elif px_std >= BLANK_STD_THRESHOLD and 0.52 <= conf <= 0.74:
            low_conf_candidates.append(entry)

    for fpath in cock_files:
        raw_px = load_and_preprocess_rgb(fpath)
        px_std = float(np.std(raw_px))
        img_int8 = quantize_pixels(raw_px, in_scale, in_zp)
        raw_out, probs = run_infer(img_int8)
        conf = float(np.max(probs))
        pred_cls = int(np.argmax(probs))
        entry = {
            "path": fpath,
            "raw_px": raw_px,
            "px_std": px_std,
            "img_int8": img_int8,
            "raw_out": raw_out,
            "probs": probs,
            "conf": conf,
            "pred_cls": pred_cls,
        }
        if px_std >= BLANK_STD_THRESHOLD and pred_cls == 1 and conf >= 0.85:
            cock_candidates.append(entry)
        elif px_std >= BLANK_STD_THRESHOLD and 0.52 <= conf <= 0.74:
            low_conf_candidates.append(entry)

    hen_candidates.sort(key=lambda x: x["conf"], reverse=True)
    cock_candidates.sort(key=lambda x: x["conf"], reverse=True)

    def pick_three(candidates):
        n = len(candidates)
        if n < 3:
            raise RuntimeError(f"Kandidat kurang dari 3 (ditemukan {n})")
        indices = [0, n // 3, (2 * n) // 3]
        return [candidates[i] for i in indices]

    selected_hens = pick_three(hen_candidates)
    selected_cocks = pick_three(cock_candidates)

    samples = []

    for idx, item in enumerate(selected_hens, start=1):
        samples.append(
            {
                "name": f"Betina_{idx:02d} (hen)",
                "img_int8": item["img_int8"],
                "raw_c0": int(item["raw_out"][0]),
                "raw_c1": int(item["raw_out"][1]),
                "expected_label": 0,
                "std_x10": int(round(item["px_std"] * 10)),
                "probs": item["probs"],
                "src": os.path.basename(item["path"]),
            }
        )

    for idx, item in enumerate(selected_cocks, start=1):
        samples.append(
            {
                "name": f"Jantan_{idx:02d} (cock)",
                "img_int8": item["img_int8"],
                "raw_c0": int(item["raw_out"][0]),
                "raw_c1": int(item["raw_out"][1]),
                "expected_label": 1,
                "std_x10": int(round(item["px_std"] * 10)),
                "probs": item["probs"],
                "src": os.path.basename(item["path"]),
            }
        )

    # Sampel #07: Blank Frame (cahaya candling terang polos tanpa telur, std = 0.0 < 12.0)
    blank_raw_px = np.full((160, 160, 3), 240.0, dtype=np.float32)
    blank_std = float(np.std(blank_raw_px))
    blank_int8 = quantize_pixels(blank_raw_px, in_scale, in_zp)
    blank_out, blank_probs = run_infer(blank_int8)
    samples.append(
        {
            "name": "Blank_Candling",
            "img_int8": blank_int8,
            "raw_c0": int(blank_out[0]),
            "raw_c1": int(blank_out[1]),
            "expected_label": 2,  # UNDEFINED karena std < 12.0
            "std_x10": int(round(blank_std * 10)),
            "probs": blank_probs,
            "src": "synthetic_blank_240",
        }
    )

    # Sampel #08: Low Confidence (< 80%) -> UNDEFINED
    if len(low_conf_candidates) > 0:
        low_conf_candidates.sort(key=lambda x: abs(x["conf"] - 0.62))
        lc = low_conf_candidates[0]
        lc_src = os.path.basename(lc["path"])
    else:
        blend_px = 0.5 * selected_hens[0]["raw_px"] + 0.5 * selected_cocks[0]["raw_px"]
        blend_std = float(np.std(blend_px))
        blend_int8 = quantize_pixels(blend_px, in_scale, in_zp)
        blend_out, blend_probs = run_infer(blend_int8)
        lc = {
            "px_std": blend_std,
            "img_int8": blend_int8,
            "raw_out": blend_out,
            "probs": blend_probs,
        }
        lc_src = "ambiguous_blend"

    samples.append(
        {
            "name": "Ambigu_LowConf",
            "img_int8": lc["img_int8"],
            "raw_c0": int(lc["raw_out"][0]),
            "raw_c1": int(lc["raw_out"][1]),
            "expected_label": 2,  # UNDEFINED karena max(probs) < 0.80
            "std_x10": int(round(lc["px_std"] * 10)),
            "probs": lc["probs"],
            "src": lc_src,
        }
    )

    print("\n=== RINGKASAN 8 GOLDEN SAMPLES ===")
    for i, s in enumerate(samples):
        lbl_str = {0: "BETINA (0)", 1: "JANTAN (1)", 2: "UNDEFINED (2)"}[s["expected_label"]]
        print(
            f"  [{i+1:02d}] {s['name']:<18} | Raw=[{s['raw_c0']:4d}, {s['raw_c1']:4d}] "
            f"| P(B)={s['probs'][0]*100:5.1f}% P(J)={s['probs'][1]*100:5.1f}% "
            f"| std={s['std_x10']/10.0:4.1f} | Target={lbl_str} | src={s['src']}"
        )

    print(f"\n[INFO] Menulis header C/C++ ke: {OUTPUT_HEADER}")
    with open(OUTPUT_HEADER, "w", encoding="utf-8") as f:
        f.write("// Auto-generated Golden Data untuk Simulasi AI PP2 (Deteksi Jenis Kelamin Telur Ayam)\n")
        f.write("// Model: MobileNetV2 (alpha=0.35) INT8 [1, 160, 160, 3]\n")
        f.write("#ifndef GOLDEN_DATA_PP2_H\n")
        f.write("#define GOLDEN_DATA_PP2_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("#define NUM_GOLDEN_SAMPLES 8\n")
        f.write("#define GOLDEN_INPUT_SIZE  76800  // 160 * 160 * 3 bytes\n\n")

        f.write("static const char* golden_sample_names[NUM_GOLDEN_SAMPLES] = {\n")
        for s in samples:
            f.write(f'    "{s["name"]}",\n')
        f.write("};\n\n")

        f.write("// Format: {ref_raw_c0, ref_raw_c1, expected_label (0=Betina, 1=Jantan, 2=Undefined), pixel_std_x10}\n")
        f.write("static const int16_t golden_output[NUM_GOLDEN_SAMPLES][4] = {\n")
        for s in samples:
            f.write(
                f"    {{{s['raw_c0']:4d}, {s['raw_c1']:4d}, {s['expected_label']}, {s['std_x10']:3d}}}, // {s['name']} ({s['src']})\n"
            )
        f.write("};\n\n")

        f.write('// 8 * 76,800 = 614,400 bytes (600 KB) ditempatkan di section ".algo" pada CM55M_S_SRAM\n')
        f.write('__attribute__((aligned(16), section(".algo"))) static const int8_t golden_input[NUM_GOLDEN_SAMPLES * GOLDEN_INPUT_SIZE] = {\n')

        all_bytes = np.concatenate([s["img_int8"].flatten() for s in samples])
        assert len(all_bytes) == 8 * 76800

        chunk_size = 32
        lines = []
        for idx in range(0, len(all_bytes), chunk_size):
            chunk = all_bytes[idx : idx + chunk_size]
            lines.append("    " + ",".join(str(int(v)) for v in chunk) + ",\n")
        f.writelines(lines)
        f.write("};\n\n")
        f.write("#endif // GOLDEN_DATA_PP2_H\n")

    size_mb = os.path.getsize(OUTPUT_HEADER) / (1024 * 1024)
    print(f"[SUCCESS] Selesai menulis golden_data.h ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()

