# Ringkasan Hasil Pelatihan & Evaluasi Model DOC Sexing AI (Notebook 4)

Dokumen ini menyajikan rangkuman komprehensif dari proses pelatihan, evaluasi, kuantisasi INT8, dan analisis performa model kecerdasan buatan (*Deep Learning*) untuk klasifikasi jenis kelamin anak ayam umur sehari (*Day-Old Chick* / DOC) menggunakan arsitektur **MobileNetV2 ($\alpha=0.35$)** pada [4_training_dan_evaluasi_mobilenetv2_doc.ipynb](file:///home/nexus_is/5.Project/Groove-Vision-Ai-v2/PP2/notebook/4_training_dan_evaluasi_mobilenetv2_doc.ipynb).

---

## 1. Ikhtisar Dataset & Pra-Pemrosesan

Model dilatih menggunakan dataset augmentasi terstratifikasi yang difokuskan pada ujung bentangan bulu sayap (*wing-tip macro crop*):

| Parameter Dataset | Keterangan |
| :--- | :--- |
| **Sumber Data Asli** | Dataset *Feather Sexing* (Roboflow: Hen / Cock) |
| **Teknik Augmentasi** | *Wing-Focused Zoom* (skala 0.38 – 0.48), rotasi halus, flip horizontal, dan *Subtle Cutout* (8–16%) |
| **Resolusi Input** | $160 \times 160 \times 3$ RGB (Sesuai spesifikasi akselerator NPU) |
| **Total Sampel** | **2.000 citra** (Seimbang 50% Betina : 50% Jantan) |
| **Pembagian Data** | • **Train Set**: 1.400 citra (700 Betina, 700 Jantan)<br>• **Validation Set**: 400 citra (200 Betina, 200 Jantan)<br>• **Test Set**: 200 citra (100 Betina, 100 Jantan) |

---

## 2. Arsitektur Model

Arsitektur dirancang khusus untuk memenuhi batasan memori SRAM (Tensor Arena $\le 450\text{ KB}$) pada mikrokontroler **Himax WiseEye2 (Cortex-M55)** dan akselerator **Arm Ethos-U55 NPU**:

```text
[Input Citra: 160x160x3 RGB]
         │
         ▼
[MobileNetV2 Backbone (alpha=0.35, ImageNet Pre-trained)]
  - 154 Layers total
         │
         ▼
[GlobalAveragePooling2D] -> 448 feature maps
         │
         ▼
[BatchNormalization + Dropout(0.35)]
         │
         ▼
[Dense(64, ReLU) + L2 Regularization (1e-4)]
         │
         ▼
[BatchNormalization + Dropout(0.20)]
         │
         ▼
[Dense(2, Softmax)] -> [P(Betina/Female), P(Jantan/Male)]
```

---

## 3. Strategi & Riwayat Pelatihan (*Two-Phase Training*)

Pelatihan dilakukan dalam dua tahap dengan mekanisme proteksi *overfitting* yang ketat:

### Fase 1: *Head Warmup* (12 Epochs)
- **Konfigurasi**: Seluruh 154 layer backbone dibekukan (*frozen*), hanya *classification head* yang dilatih. Optimizer Adam ($\text{lr} = 10^{-3}$).
- **Hasil**:
  - Epoch 1: $\text{Loss} = 0.6593, \text{Acc} = 64.29\% \;\big|\; \text{Val Loss} = 0.4364, \text{Val Acc} = 82.50\%$
  - Epoch 6: $\text{Loss} = 0.2096, \text{Acc} = 93.50\% \;\big|\; \text{Val Loss} = 0.1818, \text{Val Acc} = 95.00\%$
  - Epoch 12: $\text{Loss} = 0.1371, \text{Acc} = 95.71\% \;\big|\; \text{Val Loss} = 0.1239, \text{Val Acc} = 97.25\%$
  - *Dinamika*: $\text{Val Loss}$ menurun teratur selaras dengan $\text{Train Loss}$ tanpa ada divergensi.

### Fase 2: *Selective Fine-Tuning* (Maksimal 20 Epochs, Early Stopping)
- **Konfigurasi**: Membuka 26 layer teratas backbone (*trainable*), Optimizer Adam dengan *Cosine Decay Learning Rate* ($\text{lr} = 10^{-4} \rightarrow 5 \times 10^{-6}$), *Early Stopping* ($\text{patience} = 6$, `restore_best_weights=True`).
- **Hasil**:
  - Epoch 1: $\text{Loss} = 0.1899, \text{Acc} = 92.50\% \;\big|\; \text{Val Loss} = 0.0781, \text{Val Acc} = 98.00\%$
  - Epoch 4: $\text{Loss} = 0.0345, \text{Acc} = 99.00\% \;\big|\; \text{Val Loss} = 0.0305, \text{Val Acc} = 99.00\%$
  - Epoch 7: $\text{Loss} = 0.0113, \text{Acc} = 99.79\% \;\big|\; \text{Val Loss} = 0.0185, \text{Val Acc} = 99.25\%$
  - **Epoch 11 (Best Checkpoint)**: $\mathbf{\text{Loss} = 0.0087, \text{Acc} = 99.86\% \;\big|\; \text{Val Loss} = 0.01379, \text{Val Acc} = 99.50\%}$
  - Epoch 12–17: *Training loss* mendekati nol ($\approx 0.0036$), sedangkan *validation loss* stagnan di $0.018 - 0.026$.
  - **Epoch 17**: Terpicu *Early Stopping*. Sistem secara otomatis memulihkan bobot terbaik dari **Epoch 11** (`Restoring model weights from the end of the best epoch: 11`).

---

## 4. Hasil Evaluasi Independen (*Test Set* — 200 Citra)

Evaluasi dilakukan pada 200 citra uji yang belum pernah digunakan dalam pelatihan maupun validasi, dengan ambang batas keputusan optimal **$\text{Threshold} = 0.50$ (50.0%)**:

### Metrik Kuantitatif Utama

| Metrik Evaluasi | Nilai Float32 Model | Nilai INT8 Model (NPU) | Target Minimum | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Akurasi Keseluruhan (*Test Accuracy*)** | **100.00%** | **100.00%** | $\ge 90.0\%$ | **LULUS (Sempurna)** |
| **Macro F1-Score** | **100.00%** | **100.00%** | $\ge 90.0\%$ | **LULUS (Sempurna)** |
| **Sensitivity (Recall Jantan/Male)** | **100.00%** | **100.00%** | $\ge 90.0\%$ | **LULUS (Sempurna)** |
| **Specificity (Recall Betina/Female)** | **100.00%** | **100.00%** | $\ge 90.0\%$ | **LULUS (Sempurna)** |
| **Balanced Accuracy** | **100.00%** | **100.00%** | $\ge 90.0\%$ | **LULUS (Sempurna)** |
| **Matthews Correlation Coefficient (MCC)** | **+1.0000** | **+1.0000** | $\ge 0.80$ | **LULUS (Sempurna)** |
| **Area Under ROC Curve (ROC-AUC)** | **1.0000** | **1.0000** | $\ge 0.95$ | **LULUS (Sempurna)** |

### Laporan Klasifikasi (*Classification Report*)
```text
              Precision    Recall    F1-Score   Support
      Female     1.0000    1.0000      1.0000       100
        Male     1.0000    1.0000      1.0000       100

    Accuracy                           1.0000       200
   Macro Avg     1.0000    1.0000      1.0000       200
Weighted Avg     1.0000    1.0000      1.0000       200
```

### Matriks Konfusi (*Confusion Matrix*)
```text
                  Prediksi Betina (0)    Prediksi Jantan (1)
Aktual Betina (0)         100                     0
Aktual Jantan (1)           0                   100
```

---

## 5. Kuantisasi INT8 & Kompresi Edge AI

Model dikonversi menjadi format kuantisasi integer penuh (*Full INT8 Post-Training Quantization*) menggunakan dataset representatif:

| Karakteristik | Model Float32 (Keras) | Model INT8 (TFLite) | Efek Kompresi / Penghematan |
| :--- | :---: | :---: | :---: |
| **Ukuran Berkas (*File Size*)** | 4.524,8 KB (~4,52 MB) | **612,3 KB** | **Kompresi 7,39x lebih kecil** |
| **Format Parameter Bobot** | Float32 (32-bit floating point) | Int8 (8-bit signed integer) | Penurunan kebutuhan bandwidth memori 75% |
| **Input Tensor Quantization** | - | Scale: `0.007843`, Zero Point: `-1` | $q = \text{round}(x / 0.007843) - 1 = x - 128$ |
| **Output Tensor Quantization**| - | Scale: `0.003906`, Zero Point: `-128` | Output probabilitas $0.0 \dots 1.0$ |
| **Penurunan Akurasi (*Drop*)** | 100.00% | 100.00% | **0.00% (Tanpa degradasi akurasi)** |
| **Target Akselerasi** | CPU / GPU Laptop | **Arm Ethos-U55 NPU** | Didukung 100% operator TFLM NPU |
| **Estimasi Latensi Eksekusi** | ~85 ms (Host CPU) | **~14,57 ms (Grove Vision AI V2)** | Akselerasi hardware ~6x lebih cepat |

---

## 6. Analisis Kritis: Apakah Terjadi *Overfitting*?

### Penilaian: **TIDAK Terjadi Overfitting pada Distribusi Uji (*In-Distribution*)**

1. **Kurva Loss Konsisten**: Tidak ada fenomena *divergensi* (*val loss* melonjak naik saat *train loss* turun).
2. **Bobot Terbaik Terproteksi**: Pembobotan akhir diambil dari Epoch 11 (bukan saat over-training di akhir).
3. **Generalisasi Uji Sempurna**: Akurasi pada *validation set* (99.50%) dan *test set* (100.00%) membuktikan model mampu mengekstrak fitur garis bulu sayap bertingkat (*Female*) vs rata (*Male*) dengan sangat diskriminatif.

### Catatan Realitas Lapangan (*Domain Gap* / *Out-of-Distribution*):
* Akurasi 100% ini tercapai saat masukan adalah **foto makro jarak dekat yang berfokus pada bentangan bulu sayap**.
* Jika pengguna mengunggah foto mentah dari jarak jauh (di mana 80% citra adalah badan ayam, jari tangan, atau lantai ubin), fitur bulu sayap akan terdistorsi saat di-*downsample* ke $160 \times 160$, yang dapat memicu salah klasifikasi. Pengambilan gambar harus dipastikan berfokus pada area sayap.

---

## 7. Kesimpulan & Rekomendasi Deployment

1. **Kelayakan Model**: Model MobileNetV2 INT8 DOC Sexing dinyatakan **sangat layak dan optimal** untuk implementasi *Edge AI* pada modul Seeed Grove Vision AI V2.
2. **Parameter Firmware**:
   - `CONFIDENCE_THRESHOLD = 50.0f`
   - `BLANK_STD_THRESHOLD = 12.0f`
   - Pemetaan kelas: `0 = Betina (Female)`, `1 = Jantan (Male)`, `2 = undefined/blank`.
3. **Integrasi Sistem**: Model telah dikompilasi ke format Vela (`model_doc_int8_vela.tflite`) dan terverifikasi berjalan pada firmware `tflm_pp2_full_v2` dengan latensi NPU $\approx 14,57\text{ ms}$ (68,6 FPS).
