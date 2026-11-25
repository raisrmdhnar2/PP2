# Firmware `tflm_pp2_full_kfold` — DOC Sexing (Notebook 4a Final Model)

Firmware untuk **Seeed Grove Vision AI Module V2** (Arm Cortex-M55 + Ethos-U55 NPU) yang menjalankan inferensi klasifikasi jenis kelamin anak ayam DOC (`Female` vs `Male`) dengan model **MobileNetV2 (`alpha=0.35`) INT8** (`160×160×3`).

## Spesifikasi Model & Logika Keputusan
- **Model Sumber**: `PP2/model/model_doc_final_int8.tflite` (hasil training 100% data 31 subjek dari `notebook/4a_evaluasi_cross_validation_int8_calibrated.ipynb`).
- **Input Tensor**: `[1, 160, 160, 3] int8` (kuantisasi dinamis melalui 256-byte LUT berdasarkan `scale` & `zero_point` tensor input).
- **Output Tensor**: `[1, 2] int8` (`0: Female / Betina`, `1: Male / Jantan`).
- **Threshold Keputusan**:
  - `pixel_std < 12.0` → `Class 2: undefined` (`BLANK_FRAME`)
  - `P(Jantan) >= 50.0%` → `Class 1: Jantan (Male)` *(Default Threshold `0.50` / Argmax)*
  - `P(Jantan) < 50.0%` → `Class 0: Betina (Female)`
- **Protokol Komunikasi**:
  - **USB Serial (921600 baud)**: `PP2IP` (ping) dan `PP2IR` + `76.800` byte RGB `160×160` + `2` byte checksum.
  - **I2C Master (`PA2`/`PA3` → Slave `0x55`)**: Paket 5-byte `[0xAA, class_id, conf_int, conf_dec, checksum]`.

## 1. Kompilasi Model INT8 dengan Arm Ethos-U Vela
```bash
cd PP2/model
vela model_doc_final_int8.tflite --accelerator-config ethos-u55-64 --output-dir ./vela_out
```

## 2. Build Firmware (`EPII_CM55M_APP_S`)
Salin folder `tflm_pp2_full_kfold` ke `EPII_CM55M_APP_S/app/scenario_app/tflm_pp2_full_kfold` pada SDK `Seeed_Grove_Vision_AI_Module_V2`, lalu set di `EPII_CM55M_APP_S/makefile`:
```makefile
APP_TYPE = tflm_pp2_full_kfold
```
Lalu jalankan `make` sesuai environment Linux/Windows SDK.

## 3. Flash Firmware & Model ke Grove Vision AI V2
Alamat flash model didefinisikan di `common_config.h` (`MOBILENET_CLASSIFICATION_FLASH_ADDR = 0x3AB7B000`, offset `0xB7B000`).

```bash
python3 xmodem/xmodem_send.py \
  --port=/dev/ttyACM0 \
  --baudrate=921600 \
  --protocol=xmodem \
  --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img \
  --model="PP2/model/vela_out/model_doc_final_int8_vela.tflite 0xB7B000 0x00000"
```
Setelah selesai, tekan tombol **RESET** pada board Grove Vision AI V2.
