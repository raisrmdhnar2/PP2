# Panduan Notebook Training, Evaluasi, & Inferensi (`MobileNetV2 INT8`)

Folder ini berisi 2 *Jupyter Notebook* utama untuk merealisasikan dan menguji model **`model_int8.tflite`** yang dirancang ringan agar dapat dikompilasi **100% di NPU Arm Ethos-U55** (Grove Vision AI V2):

1. **[1_training_dan_evaluasi_mobilenetv2.ipynb](file:///home/nexus_is/PP2/notebook/1_training_dan_evaluasi_mobilenetv2.ipynb)**
   * Memuat dataset dari `../data/Sexing-Chicken-2` atau `../data/dateset` dengan penguncian urutan kelas `0 = Betina (hen)` dan `1 = Jantan (cock)`.
   * Melatih model ringan **`MobileNetV2 (alpha=0.35, 160x160x3)`** dalam 2 fase (*Feature Extraction* & *Fine-Tuning*).
   * Menampilkan evaluasi lengkap (*Accuracy/Loss curves*, *Confusion Matrix*, dan *Classification Report*).
   * Mengekspor model ke **`../model/model_int8.tflite`** dengan *static input shape* `[1, 160, 160, 3]` dan *Full Integer Quantization (`INT8`)* serta mengompilasinya dengan **Arm Vela Compiler** (`model_int8_vela.tflite`).

2. **[2_deteksi_gambar_inferensi.ipynb](file:///home/nexus_is/PP2/notebook/2_deteksi_gambar_inferensi.ipynb)**
   * Memuat **`../model/model_int8.tflite`** menggunakan `tf.lite.Interpreter`.
   * Menginput gambar uji (baik gambar tunggal maupun *batch* beberapa gambar) dan mendeteksi apakah objek termasuk **Class `0` (Betina / hen)** atau **Class `1` (Jantan / cock)**.
   * Menampilkan visualisasi gambar, grafik bar *confidence score* (%), dan simulasi aksi sudut servo ESP32 (`0°`, `180°`, atau `90° Standby`).

