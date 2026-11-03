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


def find_chick_center(img: Image.Image) -> tuple:
    """
    Mendeteksi titik pusat (center of mass) objek anak ayam (DOC) secara adaptif
    berdasarkan mask warna kuning bulu & kontras foreground terhadap latar belakang.
    Mencegah pemotongan bagian kepala/sayap pada orientasi portrait maupun landscape.
    """
    w, h = img.size
    thumb = img.resize((160, max(1, int(160 * h / w))))
    tw, th = thumb.size
    arr = np.array(thumb, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    # Filter warna kuning anak ayam (R & G tinggi, B relatif rendah)
    mask = (r > 80) & (g > 65) & (r + g > 2 * b + 15)
    if np.sum(mask) > 50:
        ys, xs = np.where(mask)
        cx = float(np.median(xs) / tw)
        cy = float(np.median(ys) / th)
        return cx, cy
    # Fallback jika mask kosong: portrait atas-tengah (0.50, 0.28), landscape tengah (0.50, 0.50)
    return 0.50, (0.28 if h > w else 0.50)


def extract_square_roi(img: Image.Image, scale: float = 0.58, shift_ratio: tuple = (0.0, 0.0), center_ratio: tuple = None) -> Image.Image:
    """
    Mengambil Region of Interest (ROI) persegi fokus tepat pada tubuh & sayap anak ayam
    tanpa memotong bulu sayap atau menyusutkan gambar dengan latar belakang meja yang lebar.
    Normalisasi orientasi EXIF tanpa rotasi paksa yang merusak orientasi asli.
    """
    w, h = img.size

    if center_ratio is not None:
        cx_ratio, cy_ratio = center_ratio
    else:
        cx_ratio, cy_ratio = find_chick_center(img)
    min_dim = min(w, h)
    crop_size = int(min_dim * scale)

    # Titik tengah adaptif dengan pergeseran halus multi-crop
    center_x = int(cx_ratio * w + shift_ratio[0] * crop_size * 0.25)
    center_y = int(cy_ratio * h + shift_ratio[1] * crop_size * 0.25)

    left = max(0, min(w - crop_size, center_x - (crop_size // 2)))
    top = max(0, min(h - crop_size, center_y - (crop_size // 2)))
    right = left + crop_size
    bottom = top + crop_size

    cropped = img.crop((left, top, right, bottom))
    return cropped.resize(TARGET_SIZE, resample=Image.LANCZOS)


def random_augment_pil(img: Image.Image, apply_erasing: bool = True) -> Image.Image:
    """
    Transformasi augmentasi lanjutan pada citra PIL 160x160
    khusus untuk meningkatkan generalisasi dan F1-score:
    - Geometric transforms (Flip, Rotate, Zoom, Jitter)
    - Photometric transforms (Brightness, Contrast, Color, Sharpness)
    - Anti-Shortcut Learning: Random Cutout / Erasing
    """
    # 1. Random Horizontal Flip (p=0.5)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. Random Slight Vertical Flip (p=0.1)
    if random.random() < 0.1:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # 3. Random Rotation (-20° s/d +20°)
    angle = random.uniform(-20.0, 20.0)
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

    # 6. Variasi Pencahayaan (Brightness 0.82 s/d 1.18)
    enhancer_b = ImageEnhance.Brightness(img)
    img = enhancer_b.enhance(random.uniform(0.82, 1.18))

    # 7. Variasi Kontras (0.85 s/d 1.25) - Penting untuk menonjolkan tekstur bulu
    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_c.enhance(random.uniform(0.85, 1.25))

    # 8. Variasi Saturasi Warna (0.85 s/d 1.15)
    enhancer_s = ImageEnhance.Color(img)
    img = enhancer_s.enhance(random.uniform(0.85, 1.15))

    # 9. Penajaman Bulu (Sharpness 1.2 s/d 1.8) atau Blur Ringan
    r = random.random()
    if r < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.6)))
    elif r < 0.6:
        enhancer_sh = ImageEnhance.Sharpness(img)
        img = enhancer_sh.enhance(random.uniform(1.2, 1.8))

    # 10. Anti-Shortcut Learning: Subtle Cutout / Erasing (p=0.25)
    # Menutup kotak kecil 8-16% ukuran citra agar tidak menutupi seluruh bulu sayap
    if apply_erasing and random.random() < 0.25:
        erase_w = int(w * random.uniform(0.08, 0.16))
        erase_h = int(h * random.uniform(0.08, 0.16))
        erase_x = random.randint(0, w - erase_w)
        erase_y = random.randint(0, h - erase_h)
        erase_patch = Image.new("RGB", (erase_w, erase_h), (random.randint(110, 140), random.randint(110, 140), random.randint(110, 140)))
        img.paste(erase_patch, (erase_x, erase_y))

    return img


def generate_augmented_dataset(
    src_dir: pathlib.Path,
    out_dir: pathlib.Path,
    target_per_class: int = 1000,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.20,
    split_by_chick: bool = False,
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
    print(f"Strategi Split     : {'Bebas Bocor Data (Chick-Level Partition)' if split_by_chick else 'Stratified Multi-Crop (All Chicks Represented)'}")
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

        # Buat sub-folder target train, valid, test dan bersihkan file lama
        for split_name in ["train", "valid", "test"]:
            split_dir = out_dir / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for old_f in split_dir.glob("*.jpg"):
                try:
                    old_f.unlink()
                except Exception:
                    pass

        if split_by_chick and len(raw_files) >= 5:
            shuffled_raw = list(raw_files)
            random.shuffle(shuffled_raw)

            # Hitung proporsi file mentah lebih seimbang
            n_raw_test = max(3, int(len(shuffled_raw) * 0.20))
            n_raw_val  = max(3, int(len(shuffled_raw) * 0.20))
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
                
                loaded_raw_imgs = []
                for rf in raw_file_list:
                    try:
                        with Image.open(rf) as im:
                            trans_im = ImageOps.exif_transpose(im).convert("RGB")
                            c_ratio = find_chick_center(trans_im)
                            loaded_raw_imgs.append((trans_im, c_ratio))
                    except Exception as e:
                        print(f"  [WARN] Gagal membaca {rf.name}: {e}")

                if not loaded_raw_imgs:
                    continue

                for j in range(target_count):
                    base_raw, c_ratio = loaded_raw_imgs[j % len(loaded_raw_imgs)]
                    
                    if split_name == "train":
                        scale_val = random.uniform(0.38, 0.48) if j >= len(loaded_raw_imgs) else 0.44
                        shift_val = (random.uniform(-0.10, 0.10), random.uniform(-0.10, 0.10)) if j >= len(loaded_raw_imgs) else (0.0, 0.0)
                        cropped_roi = extract_square_roi(base_raw, scale=scale_val, shift_ratio=shift_val, center_ratio=c_ratio)
                        aug_img = random_augment_pil(cropped_roi, apply_erasing=True) if j >= len(loaded_raw_imgs) else cropped_roi
                    else:
                        scale_val = random.uniform(0.40, 0.48) if j >= len(loaded_raw_imgs) else 0.44
                        shift_val = (random.uniform(-0.06, 0.06), random.uniform(-0.06, 0.06)) if j >= len(loaded_raw_imgs) else (0.0, 0.0)
                        cropped_roi = extract_square_roi(base_raw, scale=scale_val, shift_ratio=shift_val, center_ratio=c_ratio)
                        aug_img = random_augment_pil(cropped_roi, apply_erasing=False) if j >= len(loaded_raw_imgs) else cropped_roi

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
                        trans_im = ImageOps.exif_transpose(im).convert("RGB")
                        c_ratio = find_chick_center(trans_im)
                        loaded_raw_imgs.append((trans_im, c_ratio))
                except Exception as e:
                    print(f"  [WARN] Gagal membaca {rf.name}: {e}")

            split_assignments = (
                ["train"] * n_train +
                ["valid"] * n_valid +
                ["test"] * n_test
            )
            random.shuffle(split_assignments)

            counters = {"train": 0, "valid": 0, "test": 0}
            for i, split_name in enumerate(split_assignments):
                counters[split_name] += 1
                base_raw, c_ratio = loaded_raw_imgs[i % len(loaded_raw_imgs)]
                scale_val = random.uniform(0.38, 0.48) if i >= len(loaded_raw_imgs) else 0.44
                shift_val = (random.uniform(-0.10, 0.10), random.uniform(-0.10, 0.10)) if i >= len(loaded_raw_imgs) else (0.0, 0.0)
                cropped_roi = extract_square_roi(base_raw, scale=scale_val, shift_ratio=shift_val, center_ratio=c_ratio)

                if split_name == "train":
                    aug_img = random_augment_pil(cropped_roi, apply_erasing=True) if i >= len(loaded_raw_imgs) else cropped_roi
                else:
                    aug_img = random_augment_pil(cropped_roi, apply_erasing=False) if i >= len(loaded_raw_imgs) else cropped_roi

                if aug_img.size != TARGET_SIZE:
                    aug_img = aug_img.resize(TARGET_SIZE, resample=Image.BILINEAR)

                out_filename = f"{cls.lower()}_{split_name}_{counters[split_name]:04d}.jpg"
                out_path = out_dir / split_name / cls / out_filename
                aug_img.save(out_path, format="JPEG", quality=92)

            print(f"  -> Generated {n_train} train, {n_valid} valid, {n_test} test untuk '{cls}' (Multi-Crop dari seluruh {len(raw_files)} citra mentah)...")

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
