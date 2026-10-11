#!/usr/bin/env python3
"""
doc_augmented.py — Generator Augmentasi Dataset DOC (Day-Old Chick) Sexing
Target: Memperbanyak dataset mentah (16 Female, 15 Male) menjadi 1.000 citra per kelas.

Fitur Peningkatan Kualitas & F1-Score:
  1. Center / Aspect-Preserving Crop (ROI):
     Mencegah distorsi penyusutan ekstrem (2592x4608 -> 160x160 squash merusak ciri bulu).
     Dilakukan crop terpusat dengan aspect ratio 1:1 sebelum resize, menjaga detail bulu sayap/vent.
  2. Multi-Scale & Multi-Crop Augmentation:
     Mengambil beberapa variasi crop (scale 0.75 - 1.00) dari area kunci citra resolusi tinggi.
  3. Photometric & Geometric Diversity:
     Rotasi, jitter kecerahan/kontras, saturasi, penajaman bulu, dan flipping.
  4. Pembagian Data Seimbang (Stratified Chick-Level):
     Mencegah data leakage namun memastikan representasi train/val/test optimal.
"""

import os
import sys
import random
import argparse
import pathlib
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageChops

TARGET_SIZE = (160, 160)


def extract_square_roi(img: Image.Image, scale: float = 0.85, shift_ratio: tuple = (0.0, 0.0)) -> Image.Image:
    """
    Mengambil Region of Interest (ROI) persegi dari citra resolusi tinggi
    tanpa merusak aspect ratio (menghindari distorsi penyusutan).
    """
    w, h = img.size
    min_dim = min(w, h)
    crop_size = int(min_dim * scale)

    # Titik tengah dengan pergeseran halus
    center_x = (w // 2) + int(shift_ratio[0] * (w - crop_size) * 0.5)
    center_y = (h // 2) + int(shift_ratio[1] * (h - crop_size) * 0.5)

    left = max(0, min(w - crop_size, center_x - (crop_size // 2)))
    top = max(0, min(h - crop_size, center_y - (crop_size // 2)))
    right = left + crop_size
    bottom = top + crop_size

    cropped = img.crop((left, top, right, bottom))
    return cropped.resize(TARGET_SIZE, resample=Image.LANCZOS)


def random_augment_pil(img: Image.Image) -> Image.Image:
    """
    Transformasi augmentasi lanjutan pada citra PIL 160x160
    khusus untuk meningkatkan generalisasi dan F1-score.
    """
    # 1. Random Horizontal Flip (p=0.5)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. Random Slight Vertical Flip (p=0.1)
    if random.random() < 0.1:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # 3. Random Rotation (-25° s/d +25°)
    angle = random.uniform(-25.0, 25.0)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(128, 128, 128))

    # 4. Random Scale / Zoom (0.88 s/d 1.15)
    w, h = img.size
    scale = random.uniform(0.88, 1.15)
    new_w, new_h = int(w * scale), int(h * scale)
    if scale > 1.0:
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        img = img.crop((left, top, left + w, top + h))
    elif scale < 1.0:
        scaled_img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        img = Image.new("RGB", (w, h), (128, 128, 128))
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        img.paste(scaled_img, (paste_x, paste_y))

    # 5. Translation / Jitter Posisi (±8%)
    shift_x = int(random.uniform(-0.08, 0.08) * w)
    shift_y = int(random.uniform(-0.08, 0.08) * h)
    if shift_x != 0 or shift_y != 0:
        img = ImageChops.offset(img, shift_x, shift_y)

    # 6. Variasi Pencahayaan (Brightness 0.80 s/d 1.20)
    enhancer_b = ImageEnhance.Brightness(img)
    img = enhancer_b.enhance(random.uniform(0.80, 1.20))

    # 7. Variasi Kontras (0.80 s/d 1.25) - Penting untuk menonjolkan tekstur bulu
    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_c.enhance(random.uniform(0.80, 1.25))

    # 8. Variasi Saturasi Warna (0.85 s/d 1.15)
    enhancer_s = ImageEnhance.Color(img)
    img = enhancer_s.enhance(random.uniform(0.85, 1.15))

    # 9. Penajaman Bulu (Sharpness 1.2 s/d 1.8) atau Blur Ringan
    r = random.random()
    if r < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))
    elif r < 0.5:
        enhancer_sh = ImageEnhance.Sharpness(img)
        img = enhancer_sh.enhance(random.uniform(1.2, 1.8))

    return img


def generate_augmented_dataset(
    src_dir: pathlib.Path,
    out_dir: pathlib.Path,
    target_per_class: int = 1000,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    split_by_chick: bool = True,
    seed: int = 42,
):
    """Menghasilkan dataset augmented terstandarisasi untuk kelas Female dan Male."""
    random.seed(seed)
    np.random.seed(seed)

    classes = ["Female", "Male"]
    n_train = int(target_per_class * train_ratio)
    n_valid = int(target_per_class * valid_ratio)
    n_test = target_per_class - n_train - n_valid

    print("=" * 70)
    print("  AUGMENTASI DATASET DOC SEXING (DENGAN ROI PRESERVATION & CROP)")
    print("=" * 70)
    print(f"Direktori Sumber   : {src_dir.resolve()}")
    print(f"Direktori Output   : {out_dir.resolve()}")
    print(f"Target per Kelas   : {target_per_class} citra")
    print(f"Proporsi Pembagian : Train={n_train} | Valid={n_valid} | Test={n_test}")
    print(f"Strategi Split     : {'Bebas Bocor Data (Chick-Level Partition)' if split_by_chick else 'Stratified Random'}")
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

        if split_by_chick and len(raw_files) >= 5:
            shuffled_raw = list(raw_files)
            random.shuffle(shuffled_raw)

            # Hitung proporsi file mentah (misal: 16 file -> 12 train, 2 valid, 2 test)
            n_raw_test = max(2, int(len(shuffled_raw) * 0.15))
            n_raw_val  = max(2, int(len(shuffled_raw) * 0.15))
            n_raw_train = len(shuffled_raw) - n_raw_val - n_raw_test

            train_raw_files = shuffled_raw[:n_raw_train]
            val_raw_files   = shuffled_raw[n_raw_train : n_raw_train + n_raw_val]
            test_raw_files  = shuffled_raw[n_raw_train + n_raw_val :]

            print(f"  - Alokasi File Mentah: Train={len(train_raw_files)} | Valid={len(val_raw_files)} | Test={len(test_raw_files)}")

            split_packs = [
                ("train", train_raw_files, n_train),
                ("valid", val_raw_files,   n_valid),
                ("test",  test_raw_files,  n_test),
            ]

            for split_name, raw_file_list, target_count in split_packs:
                if not raw_file_list:
                    continue
                
                # Buka semua citra mentah untuk split ini
                loaded_raw_imgs = []
                for rf in raw_file_list:
                    try:
                        with Image.open(rf) as im:
                            loaded_raw_imgs.append(im.convert("RGB"))
                    except Exception as e:
                        print(f"  [WARN] Gagal membaca {rf.name}: {e}")

                if not loaded_raw_imgs:
                    continue

                for j in range(target_count):
                    base_raw = loaded_raw_imgs[j % len(loaded_raw_imgs)]
                    
                    # Ambil ROI persegi dengan variasi skala crop agar mencakup sayap & tubuh
                    scale_val = random.uniform(0.70, 0.95) if j >= len(loaded_raw_imgs) else 0.85
                    shift_val = (random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15)) if j >= len(loaded_raw_imgs) else (0.0, 0.0)
                    
                    cropped_roi = extract_square_roi(base_raw, scale=scale_val, shift_ratio=shift_val)
                    
                    if j < len(loaded_raw_imgs):
                        aug_img = cropped_roi
                    else:
                        aug_img = random_augment_pil(cropped_roi)

                    if aug_img.size != TARGET_SIZE:
                        aug_img = aug_img.resize(TARGET_SIZE, resample=Image.BILINEAR)

                    out_filename = f"{cls.lower()}_{split_name}_{j+1:04d}.jpg"
                    out_path = out_dir / split_name / cls / out_filename
                    aug_img.save(out_path, format="JPEG", quality=92)

                print(f"  -> Generated {target_count} citra untuk '{split_name}/{cls}'...")
        else:
            loaded_raw_imgs = []
            for rf in raw_files:
                try:
                    with Image.open(rf) as im:
                        loaded_raw_imgs.append(im.convert("RGB"))
                except Exception as e:
                    print(f"  [WARN] Gagal membaca {rf.name}: {e}")

            split_assignments = (
                ["train"] * n_train +
                ["valid"] * n_valid +
                ["test"] * n_test
            )
            random.shuffle(split_assignments)

            for i, split_name in enumerate(split_assignments):
                base_raw = loaded_raw_imgs[i % len(loaded_raw_imgs)]
                scale_val = random.uniform(0.70, 0.95) if i >= len(loaded_raw_imgs) else 0.85
                shift_val = (random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15)) if i >= len(loaded_raw_imgs) else (0.0, 0.0)
                cropped_roi = extract_square_roi(base_raw, scale=scale_val, shift_ratio=shift_val)

                if i < len(loaded_raw_imgs):
                    aug_img = cropped_roi
                else:
                    aug_img = random_augment_pil(cropped_roi)

                if aug_img.size != TARGET_SIZE:
                    aug_img = aug_img.resize(TARGET_SIZE, resample=Image.BILINEAR)

                out_filename = f"{cls.lower()}_{i+1:04d}.jpg"
                out_path = out_dir / split_name / cls / out_filename
                aug_img.save(out_path, format="JPEG", quality=92)

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

    script_dir = pathlib.Path(__file__).resolve().parent
    default_src = script_dir / "doc"
    default_out = script_dir / "doc_augmented"

    parser.add_argument("--src-dir", type=pathlib.Path, default=default_src, help="Path ke data mentah (default: data/doc)")
    parser.add_argument("--out-dir", type=pathlib.Path, default=default_out, help="Path output (default: data/doc_augmented)")
    parser.add_argument("--target-per-class", type=int, default=1000, help="Jumlah total gambar per kelas (default: 1000)")
    parser.add_argument("--split-random", action="store_true", help="Gunakan split acak biasa (default: Chick-Level Partition)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed deterministik")

    args = parser.parse_args()
    generate_augmented_dataset(
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        target_per_class=args.target_per_class,
        split_by_chick=not args.split_random,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
