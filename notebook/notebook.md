# Panduan Notebook Training, Evaluasi, & Inferensi (`MobileNetV2 INT8`)

Folder ini berisi *Jupyter Notebooks* untuk merealisasikan model TinyML yang dapat dikompilasi **100% di NPU Arm Ethos-U55** (Grove Vision AI V2):

---

## A. Jalur Model 1: Deteksi Jenis Kelamin Telur Ayam (*Egg Sexing*)
1. **[`1_training_dan_evaluasi_mobilenetv2.ipynb`](file:///home/nexus_is/5.Project/Groove-Vision-Ai-v2/PP2/notebook/1_training_dan_evaluasi_mobilenetv2.ipynb)**
   * Memuat dataset dari `../data/Sexing-Chicken-2` (`0 = Betina / hen`, `1 = Jantan / cock`).
   * Melatih model ringan **`MobileNetV2 (alpha=0.35, 160x160x3)`** dalam 2 fase (*Feature Extraction* & *Fine-Tuning*).
   * Mengekspor model ke `../model/model_int8.tflite` (Full Integer Quantization) dan kompilasi Vela NPU `model_int8_vela.tflite`.
2. **[`2_deteksi_gambar_inferensi.ipynb`](file:///home/nexus_is/5.Project/Groove-Vision-Ai-v2/PP2/notebook/2_deteksi_gambar_inferensi.ipynb)**
   * Pengujian inferensi lokal TFLite Interpreter untuk citra telur dengan dual-threshold (Confidence >= 80% & std >= 12.0).
3. **[`3_simulasi_hardware_grove_npu.ipynb`](file:///home/nexus_is/5.Project/Groove-Vision-Ai-v2/PP2/notebook/3_simulasi_hardware_grove_npu.ipynb)**
   * Pengiriman citra telur via serial USB-C (`/dev/ttyACM0` @ 921600 baud) ke hardware Grove Vision AI V2.

---

## B. Jalur Model 2: Deteksi Jenis Kelamin Anak Ayam (*Day-Old Chick / DOC Sexing*)
4. **[`4_training_dan_evaluasi_mobilenetv2_doc.ipynb`](file:///home/nexus_is/5.Project/Groove-Vision-Ai-v2/PP2/notebook/4_training_dan_evaluasi_mobilenetv2_doc.ipynb)**
   * Memuat dataset DOC (`data/doc_augmented/`, dengan auto-augmentasi ke 1.000 citra/kelas jika belum ada).
   * Melatih model **MobileNetV2 ($\alpha=0.35$, $160\times160\times3$)** dengan pemetaan `0 = Female` dan `1 = Male`.
   * Mengekspor model ke `../model/model_doc_int8.tflite` dan kompilasi Vela NPU `model_doc_int8_vela.tflite`.
5. **[`5_deteksi_gambar_inferensi_doc.ipynb`](file:///home/nexus_is/5.Project/Groove-Vision-Ai-v2/PP2/notebook/5_deteksi_gambar_inferensi_doc.ipynb)**
   * Pengujian inferensi lokal TFLite Interpreter untuk citra DOC dengan logika keputusan 3-State (`Female 0°`, `Male 180°`, `Undefined 90°`).
6. **[`6_simulasi_hardware_grove_npu_doc.ipynb`](file:///home/nexus_is/5.Project/Groove-Vision-Ai-v2/PP2/notebook/6_simulasi_hardware_grove_npu_doc.ipynb)**
   * Pengiriman citra DOC via USB-C serial protokol `PP2IR` (76.800 B) ke hardware Grove Vision AI V2 on-device NPU inference.
