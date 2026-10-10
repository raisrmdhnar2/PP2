# Konsep Training, Evaluasi, Kuantisasi INT8, dan Inferensi Model (MobileNetV2 - Edge NPU)

## 1. Tujuan Dokumen
Dokumen ini merangkum konsep teknis untuk merealisasikan model **`model_int8.tflite`** yang sangat ringan dan dapat dikompilasi **100% pada NPU Arm Ethos-U55** (Grove Vision AI V2), mulai dari pengolahan dataset, arsitektur model, proses *training* & evaluasi, kuantisasi penuh ke `INT8`, hingga pengujian deteksi gambar (*Class `0`* dan *Class `1`*).

---

## 2. Struktur Dataset & Pemetaan Kelas (Class Mapping)
Dataset tersedia pada direktori proyek (`data/dateset` dan `data/Sexing-Chicken-2`) yang terdiri dari dua folder kelas:
*   `hen` (Ayam Betina)
*   `cock` (Ayam Jantan)

### Aturan Kunci Pemetaan Kelas
Sesuai dengan [konsep_sistem.md](file:///home/nexus_is/PP2/docs/konsep_sistem.md), output biner sistem didefinisikan sebagai:
*   **Class `0` : Betina (`hen`)** $\rightarrow$ Mengarahkan Servo ke **`0°`**
*   **Class `1` : Jantan (`cock`)** $\rightarrow$ Mengarahkan Servo ke **`180°`**

> **Catatan Penting Implementasi:** Secara alfabetis, folder `cock` berada sebelum `hen`. Jika menggunakan *dataloader* standar tanpa konfigurasi eksplisit, `cock` akan menjadi indeks `0`. Oleh karena itu, parameter `class_names=['hen', 'cock']` **wajib dikunci secara eksplisit** saat memuat dataset agar indeks `0 = hen (Betina)` dan `1 = cock (Jantan)`.

---

## 3. Desain Arsitektur Ringan untuk 100% Kompilasi NPU
Agar model dapat dieksekusi **100% di atas NPU Arm Ethos-U55** tanpa *fallback* ke CPU (0% CPU operators pada Arm Vela), arsitektur model harus memenuhi kriteria berikut:

1.  **Menggunakan `MobileNetV2` dengan `alpha=0.35`:**
    *   *Width multiplier* `0.35` memangkas jumlah parameter secara signifikan sehingga ukuran akhir `model_int8.tflite` hanya berkisar **~450 KB – 600 KB**.
    *   Ukuran ini memastikan seluruh bobot (*weights*) dan *tensor arena* muat sepenuhnya di dalam *Internal SRAM* Grove Vision AI V2, menghasilkan *frame rate* (FPS) maksimal dan latensi sangat rendah.
2.  **Resolusi Input `160x160x3` (atau `192x192x3`):**
    *   Kompatibel dengan *pretrained weights* ImageNet pada Keras dan cukup tajam untuk mengenali ciri morfologi ayam, namun jauh lebih hemat memori dibandingkan `224x224`.
3.  **Grafik Model Bersih (*Zero Preprocessing Inside Model Graph*):**
    *   **Jangan** memasukkan layer augmentasi (`RandomFlip`, `RandomRotation`) atau layer normalisasi kompleks ke dalam arsitektur model yang akan diekspor ke TFLite, karena layer tersebut sering menghasilkan operasi `CAST`/`DEQUANTIZE` yang memaksa eksekusi turun ke CPU.
    *   Seluruh proses *resizing*, *augmentasi*, dan *normalisasi* dilakukan di luar grafik model (pada pipeline `tf.data.Dataset`).
4.  **Susunan Layer yang 100% Didukung Ethos-U55:**
    *   `Input(shape=(160, 160, 3))`
    *   `MobileNetV2(alpha=0.35, include_top=False, weights='imagenet')` $\rightarrow$ Hanya berisi `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, dan `RELU6`.
    *   `GlobalAveragePooling2D()` $\rightarrow$ Diterjemahkan menjadi operator `MEAN` di NPU.
    *   `Dropout(0.2)` $\rightarrow$ Hanya aktif saat *training*, otomatis hilang saat inferensi TFLite.
    *   `Dense(2, activation='softmax')` $\rightarrow$ Diterjemahkan menjadi `FULLY_CONNECTED` + `SOFTMAX` dengan 2 neuron output (`[prob_class_0, prob_class_1]`) yang kompatibel dengan format klasifikasi firmware Grove Vision AI V2 (SenseCraft / SSCMA).
5.  **Static Input Shape (`[1, 160, 160, 3]`) saat Ekspor TFLite:**
    *   Saat mengekspor model Keras ke TFLite, dimensi *batch* dinamis (`None`) sering memicu munculnya operator `SHAPE`, `STRIDED_SLICE`, dan `PACK` yang tidak didukung NPU dan terpaksa dijalankan di CPU.
    *   Dengan mengunci dimensi input menjadi statis `[1, 160, 160, 3]` menggunakan `tf.function` (*ConcreteFunction*) sebelum kuantisasi, seluruh operasi penghitungan dimensi dinamis hilang sehingga **100% operator dieksekusi di NPU Ethos-U55 (0% CPU fallback)**.

---

## 4. Strategi Training & Evaluasi Model

### A. Tahap Training (2 Fase)
1.  **Fase 1 – Feature Extraction (Head Training):**
    *   Membekukan (*freeze*) bobot dasar `MobileNetV2` dan hanya melatih layer klasifikasi akhir (`Dense`) selama beberapa *epoch* menggunakan `Adam(learning_rate=1e-3)`.
2.  **Fase 2 – Fine-Tuning:**
    *   Membuka (*unfreeze*) layer bagian atas `MobileNetV2` (kecuali `BatchNormalization` agar statistik tetap stabil) dan melatih ulang dengan *learning rate* kecil (`1e-4` hingga `1e-5`) menggunakan `EarlyStopping` dan `ReduceLROnPlateau`.

### B. Metrik Evaluasi & Validasi
Evaluasi dilakukan secara menyeluruh baik sebelum maupun sesudah kuantisasi:
1.  **Learning Curves:** Grafik *Training vs Validation Accuracy* dan *Loss* per *epoch* untuk mendeteksi *overfitting*.
2.  **Confusion Matrix & Classification Report:** Menghitung *Precision*, *Recall*, dan *F1-Score* untuk masing-masing kelas (`0: Betina`, `1: Jantan`) pada dataset validasi/uji.
3.  **Evaluasi Akurasi `model_int8.tflite`:** Menjalankan ulang dataset validasi/uji menggunakan `tf.lite.Interpreter` pada model `INT8` untuk memastikan tidak terjadi penurunan akurasi yang signifikan (*accuracy drop* minimal) dibandingkan model *Float32*.

---

## 5. Kuantisasi Full INT8 & Kompilasi Arm Vela

### A. Post-Training Full Integer Quantization (`INT8`)
Grove Vision AI V2 mewajibkan seluruh bobot, aktivasi, serta tensor input dan output berada dalam format integer 8-bit (`int8`):
*   **Representative Dataset:** Mengambil 100–200 gambar sampel dari dataset *training* untuk mengkalibrasi nilai *scale* dan *zero-point* pada setiap layer aktivasi.
*   **Konfigurasi Converter:**
    *   `converter.optimizations = [tf.lite.Optimize.DEFAULT]`
    *   `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]`
    *   `converter.inference_input_type = tf.int8`
    *   `converter.inference_output_type = tf.int8`
*   Output disimpan ke **`model/model_int8.tflite`**.

### B. Kompilasi NPU dengan Arm Vela
File `model_int8.tflite` dikompilasi menggunakan **Arm Vela Compiler** dengan target `ethos-u55-64`:
*   Menghasilkan file **`model/model_int8_vela.tflite`** yang siap di-*flash* ke Grove Vision AI V2.
*   Laporan Vela (*Vela Summary Report*) digunakan untuk memverifikasi bahwa **100% operator berjalan di NPU**.

---

## 6. Alur Notebook Deteksi & Inferensi Gambar
Notebook kedua dibuat khusus untuk menguji coba model `model_int8.tflite` menggunakan input gambar tunggal maupun beberapa gambar uji sekaligus:
1.  **Input Gambar:** Memilih path gambar ayam dari folder uji atau mengunggah gambar baru.
2.  **Preprocessing & Quantization Input:**
    *   Mengubah ukuran gambar ke `160x160` RGB dan menormalisasi nilai piksel.
    *   Mengonversi nilai *float* ke tipe `int8` menggunakan `input_scale` dan `input_zero_point` dari metadata tensor `model_int8.tflite`.
3.  **Inferensi & Dequantization Output:**
    *   Menjalankan `interpreter.invoke()` dan mengembalikan nilai output `int8` ke skala probabilitas (`0% - 100%`).
4.  **Keputusan & Simulasi Aktuator Servo:**
    *   Menampilkan gambar beserta hasil deteksi:
        *   **Class `0` (Betina / Hen)** atau **Class `1` (Jantan / Cock)**
        *   **Confidence Score (%)**
    *   Menerapkan *Logic Threshold* (misal `>= 80%`):
        *   Jika *confidence* $\ge 80\%$ dan prediksi `Class 0` $\rightarrow$ **Servo ke `0°` (Jalur Betina)**.
        *   Jika *confidence* $\ge 80\%$ dan prediksi `Class 1` $\rightarrow$ **Servo ke `180°` (Jalur Jantan)**.
        *   Jika *confidence* $< 80\%$ $\rightarrow$ **Servo tetap di `90°` (Standby / Abaikan)**.
