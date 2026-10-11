#!/usr/bin/env python3
"""
doc_augmented.py — Skrip Generator Augmentasi Dataset DOC (Day-Old Chick) Sexing
Target: Memperbanyak dataset mentah (16 Female, 15 Male) menjadi 1.000 citra per kelas.

Struktur Output:
  data/doc_augmented/
    ├── train/ (70% = 700 Female, 700 Male)
    ├── valid/ (15% = 150 Female, 150 Male)
    └── test/  (15% = 150 Female, 150 Male)

Cara Menjalankan:
  1. Langsung (default path relatif):
       python doc_augmented.py
  2. Kustomisasi target per kelas:
       python doc_augmented.py --target-per-class 1000
"""

import os
import sys
import random
import argparse
import pathlib
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Resolusi input model TinyML Ethos-U55
TARGET_SIZE = (160, 160)


def random_augment_pil(img: Image.Image) -> Image.Image:
    """Melakukan rangkaian transformasi fotometrik dan geometrik acak pada citra PIL."""
    # 1. Random Horizontal Flip (p=0.5)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. Random Slight Vertical Flip (p=0.1)
    if random.random() < 0.1:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # 3. Random Rotation (-22° s/d +22°)
    angle = random.uniform(-22.0, 22.0)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(128, 128, 128))

    # 4. Random Crop / Zoom (scale: 0.85 s/d 1.15)
    w, h = img.size
    scale = random.uniform(0.85, 1.15)
    new_w, new_h = int(w * scale), int(h * scale)
    if scale > 1.0:
        # Crop bagian tengah
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        img = img.crop((left, top, left + w, top + h))
    elif scale < 1.0:
        # Padding
        scaled_img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        img = Image.new("RGB", (w, h), (128, 128, 128))
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        img.paste(scaled_img, (paste_x, paste_y))

    # 5. Random Translation / Jitter Posisi (±8%)
    shift_x = int(random.uniform(-0.08, 0.08) * w)
    shift_y = int(random.uniform(-0.08, 0.08) * h)
    if shift_x != 0 or shift_y != 0:
        img = ImageOps.offset(img, shift_x, shift_y)

    # 6. Random Brightness (0.85 s/d 1.15)
    enhancer_b = ImageEnhance.Brightness(img)
    img = enhancer_b.enhance(random.uniform(0.85, 1.15))

    # 7. Random Contrast (0.85 s/d 1.15)
    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_c.enhance(random.uniform(0.85, 1.15))

    # 8. Random Color / Saturation (0.88 s/d 1.12)
    enhancer_s = ImageEnhance.Color(img)
    img = enhancer_s.enhance(random.uniform(0.88, 1.12))

    # 9. Random Sharpness / Blur Ringan (p=0.3)
    r = random.random()
    if r < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    elif r < 0.4:
        enhancer_sh = ImageEnhance.Sharpness(img)
        img = enhancer_sh.enhance(random.uniform(1.2, 1.6))

    return img


def generate_augmented_dataset(
    src_dir: pathlib.Path,
    out_dir: pathlib.Path,
    target_per_class: int = 1000,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    seed: int = 42,
):
    """Menghasilkan dataset augmented terstandarisasi untuk kelas Female dan Male."""
    random.seed(seed)
    np.random.seed(seed)

    test_ratio = 1.0 - (train_ratio + valid_ratio)
    n_train = int(target_per_class * train_ratio)
    n_valid = int(target_per_class * valid_ratio)
    n_test = target_per_class - n_train - n_valid

    classes = ["Female", "Male"]
    print("=" * 70)
    print("  AUGMENTASI DATASET DOC SEXING (DAY-OLD CHICK)")
    print("=" * 70)
    print(f"Direktori Sumber   : {src_dir.resolve()}")
    print(f"Direktori Output   : {out_dir.resolve()}")
    print(f"Target per Kelas   : {target_per_class} citra")
    print(f"Proporsi Pembagian : Train={n_train} | Valid={n_valid} | Test={n_test}")
    print("=" * 70)

    for cls in classes:
        cls_src = src_dir / cls
        if not cls_src.exists():
            print(f"[ERROR] Folder sumber tidak ditemukan: {cls_src}")
            continue

        raw_files = sorted(
            [f for f in cls_src.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )
        if not raw_files:
            print(f"[ERROR] Tidak ada citra di {cls_src}")
            continue

        print(f"\n[INFO] Memproses Kelas '{cls}': Ditemukan {len(raw_files)} citra mentah.")

        # Buat sub-folder target train, valid, test
        for split_name in ["train", "valid", "test"]:
            split_dir = out_dir / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)

        # Pre-load dan resize citra asli ke target_size
        base_images = []
        for rf in raw_files:
            try:
                with Image.open(rf) as im:
                    im_rgb = im.convert("RGB")
                    im_resized = im_rgb.resize(TARGET_SIZE, resample=Image.LANCZOS)
                    base_images.append(im_resized)
            except Exception as e:
                print(f"  [WARN] Gagal membaca {rf.name}: {e}")

        if not base_images:
            continue

        # Alokasi assignment split
        split_assignments = (
            ["train"] * n_train +
            ["valid"] * n_valid +
            ["test"] * n_test
        )
        random.shuffle(split_assignments)

        for i, split_name in enumerate(split_assignments):
            # Ambil citra basis secara round-robin atau random
            base_img = base_images[i % len(base_images)]

            # Jika indeks < jumlah citra asli, simpan citra original yang di-resize
            if i < len(base_images):
                aug_img = base_img.copy()
            else:
                aug_img = random_augment_pil(base_img)

            # Pastikan ukuran persis TARGET_SIZE
            if aug_img.size != TARGET_SIZE:
                aug_img = aug_img.resize(TARGET_SIZE, resample=Image.BILINEAR)

            out_filename = f"{cls.lower()}_{i+1:04d}.jpg"
            out_path = out_dir / split_name / cls / out_filename
            aug_img.save(out_path, format="JPEG", quality=92)

            if (i + 1) % 200 == 0 or (i + 1) == target_per_class:
                print(f"  -> Generated {i+1}/{target_per_class} citra untuk '{cls}'...")

        print(f"[OK] Selesai kelas '{cls}': {target_per_class} citra tersimpan di {out_dir}")

    print("\n" + "=" * 70)
    print("  RINGKASAN DATASET AUGMENTED")
    print("=" * 70)
    for split_name in ["train", "valid", "test"]:
        f_count = len(list((out_dir / split_name / "Female").glob("*.jpg"))) if (out_dir / split_name / "Female").exists() else 0
        m_count = len(list((out_dir / split_name / "Male").glob("*.jpg"))) if (out_dir / split_name / "Male").exists() else 0
        print(f"  - {split_name.upper():<6}: Female = {f_count:>4} | Male = {m_count:>4} | Total = {f_count + m_count:>4}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="doc_augmented.py — Generator Augmentasi Dataset DOC Sexing (1000/kelas)")

    # Default path relatif terhadap lokasi file script ini
    script_dir = pathlib.Path(__file__).resolve().parent
    default_src = script_dir / "doc"
    default_out = script_dir / "doc_augmented"

    parser.add_argument("--src-dir", type=pathlib.Path, default=default_src, help="Path ke data mentah (default: data/doc)")
    parser.add_argument("--out-dir", type=pathlib.Path, default=default_out, help="Path output (default: data/doc_augmented)")
    parser.add_argument("--target-per-class", type=int, default=1000, help="Jumlah total gambar per kelas (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed deterministik")

    args = parser.parse_args()
    generate_augmented_dataset(
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        target_per_class=args.target_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

