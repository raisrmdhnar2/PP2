# Laporan Evaluasi Kritis & Audit Validitas Ilmiah: Akurasi 100% pada Notebook 4

**Judul Dokumen**: Laporan Investigasi Ilmiah Data Leakage, Generalisasi Model, dan Realitas Akurasi 100% DOC Sexing AI  
**Berkas Terkait**: [PP2/notebook/4_training_dan_evaluasi_mobilenetv2_doc.ipynb](file:///home/nexus_is/5.Project/Groove-Vision-Ai-v2/PP2/notebook/4_training_dan_evaluasi_mobilenetv2_doc.ipynb) & [PP2/data/doc_augmented.py](file:///home/nexus_is/5.Project/Groove-Vision-Ai-v2/PP2/data/doc_augmented.py)  
**Tanggal**: 20 November 2026  
**Status**: Temuan Kritis (*Scientific Red Flag Investigation*)

---

## 1. Pernyataan Masalah & Skeptisisme Ilmiah

Dalam disiplin *Machine Learning* dan *Computer Vision* terapan, perolehan metrik:
* **Test Accuracy**: `100.00%`
* **Macro F1-Score**: `100.00%`
* **Sensitivity & Specificity**: `100.00%`
* **ROC-AUC**: `1.0000`
* **INT8 Accuracy Drop**: `0.00%`

merupakan **sinyal bahaya (*Red Flag*)** yang hampir selalu mengindikasikan adanya **kebocoran data (*Data Leakage*)**, **distribusi data yang tertutup (*In-Distribution Memorization*)**, atau **pembagian dataset yang tidak independen**, bukan generalisasi biologis yang sempurna di dunia nyata.

Kecurigaan ini terkonfirmasi ketika model diuji dengan citra anak ayam (DOC) baru di luar dataset, di mana model gagal membedakan jantan-betina secara konsisten dan cenderung mengalami bias ke kelas Betina.

---

## 2. Temuan Investigasi: Akar Masalah Mengapa Akurasi Menjadi 100%

Investigasi mendalam pada kode sumber dan struktur data menemukan **3 akar masalah fundamental**:

### A. Kebocoran Subjek (*Subject-Level Data Leakage*) — **Penyebab Utama**
Setelah memeriksa direktori data mentah [PP2/data/doc/](file:///home/nexus_is/5.Project/Groove-Vision-Ai-v2/PP2/data/doc/):
* **Jumlah Subjek Anak Ayam Mentah Hanya 31 Ekor**:
  - Kelas Betina (*Female*): **16 foto** (`Female01.jpg` s/d `Female16.jpg`)
  - Kelas Jantan (*Male*): **15 foto** (`Male01.jpg` s/d `Male14.jpg` + `Male06(1).jpg`)
* Dari 31 foto mentah beresolusi tinggi tersebut, generator augmentasi (`doc_augmented.py`) mengekstrak **2.000 citra** (1.000 per kelas) dengan strategi *Multi-Crop* dan pergeseran koordinat (*shift/scale*).
* Pada baris 278–288 [PP2/data/doc_augmented.py](file:///home/nexus_is/5.Project/Groove-Vision-Ai-v2/PP2/data/doc_augmented.py#L278-L288):
  ```python
  split_assignments = ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test
  random.shuffle(split_assignments)
  for i, split_name in enumerate(split_assignments):
      base_raw, c_ratio = loaded_raw_imgs[i % len(loaded_raw_imgs)]
      # ... multi-crop dihasilkan dan disebar acak ke train, valid, test ...
  ```
* **Dampak Langsung**:
  Potongan crop dari **ekor ayam yang persis sama** (misalnya `Female01.jpg`) dialokasikan ke dalam **Train Set** (45 potongan), **Validation Set** (13 potongan), dan **Test Set** (6 potongan).
* **Konsekuensi Ilmiah**:
  Ketika model dievaluasi pada *Test Set*, model **tidak sedang mengidentifikasi jenis kelamin anak ayam baru**, melainkan hanya mengenali kembali anak ayam yang sama persis (dengan warna bulu, sudut kamera, pencahayaan, dan latar belakang yang identik) yang sudah dipelajari ribuan kali selama proses *training*. Inilah yang menyebabkan akurasinya menjadi **100% sempurna**.

---

### B. Jebakan Fitur Pintas Latar Belakang (*Clever Hans Effect / Shortcut Learning*)
Karena hanya ada 16 ayam betina dan 15 ayam jantan:
1. **Fitur Tangan & Jari Pemeriksa**:
   Foto mentah diambil dengan tangan manusia yang memegang anak ayam. Setiap sesi pemotretan memiliki bentuk jari, warna kulit, atau posisi pegangan yang sedikit berbeda. Model deep learning (MobileNetV2) dengan mudah mempelajari korelasi semu (*spurious correlation*) antara bentuk tangan tertentu dengan label kelas, alih-alih fitur biologis bulu sayap.
2. **Karakteristik Meja, Pencahayaan, dan Bayangan**:
   Ayam betina dan ayam jantan mungkin difoto pada sesi waktu atau sudut meja yang sedikit bergeser. Model mampu memanfaatkan artefak latar belakang mikro untuk menebak kelas dengan probabilitas 100%.

---

### C. Jurang Distribusi Data (*Severe Out-of-Distribution / Domain Gap*)
Ketika Anda mengambil foto baru di lapangan dengan kamera HP:
* Tangan yang memegang berbeda.
* Jarak foto lebih jauh (seluruh badan ayam, bukan makro sayap).
* Pencahayaan dan warna kuning bulu berbeda dari 31 ayam di laboratorium.
* **Hasilnya**: Model kehilangan seluruh "fitur jalan pintas" yang dihafalnya dari 31 ayam tersebut. Karena ketidakpastian tinggi, keluaran Softmax menjadi netral ($\approx 50\%$). Karena aturan ambang batas keputusan `P(Betina) >= P(Jantan)` berada di 50.0%, model otomatis jatuh ke nilai *default* yaitu **Betina (Female)**.

---

## 3. Matriks Perbandingan: Klaim Evaluasi vs Realitas Lapangan

| Parameter Uji | Hasil Evaluasi Notebook 4 | Realitas Pengujian Lapangan | Penjelasan Ilmiah |
| :--- | :---: | :---: | :--- |
| **Akurasi (*Accuracy*)** | `100.00%` | Turun drastis pada foto baru | Terjadi *Subject Memorization*, bukan *Species Generalization*. |
| **Independensi Test Set** | Sangat Rendah | Data benar-benar independen | Test set di Notebook 4 terkontaminasi subjek yang sama dengan Train set (*Data Leakage*). |
| **Jumlah Subjek Unik** | 31 ekor ayam | Tidak terbatas di peternakan | 31 ekor tidak cukup untuk menangkap variasi biologis ras ayam di dunia nyata. |
| **Fokus Area Citra** | Crop makro terstandarisasi | Foto bebas / seluruh badan | Model tidak memiliki mekanisme deteksi objek (*Object Detection*) untuk melokalisasi sayap secara mandiri. |

---

## 4. Rekomendasi Solusi & Rencana Tindak Lanjut (*Action Plan*)

Untuk menghasilkan model yang benar-benar kredibel, dapat dipertanggungjawabkan dalam laporan resmi/sidang, serta bekerja andal di lapangan, berikut langkah perbaikan yang direkomendasikan:

### 1. Pembagian Data Berbasis Subjek Murni (*Strict Chick-Level Partitioning*)
* Pisahkan 31 foto mentah **sebelum** proses augmentasi dilakukan:
  - **Train Set**: 21 ekor ayam (11 Betina, 10 Jantan)
  - **Validation Set**: 4 ekor ayam (2 Betina, 2 Jantan)
  - **Test Set**: 6 ekor ayam (3 Betina, 3 Jantan)
* Pastikan tidak ada satu pun potongan crop dari 6 ayam di *Test Set* yang pernah dilihat oleh model saat *training*.
* **Ekspektasi Realistis**: Akurasi pada *Test Set* yang benar-benar independen kemungkinan akan berada di rentang **$78\% - 88\%$**. Angka ini jauh lebih ilmiah, valid, dan jujur untuk dilaporkan.

### 2. Evaluasi Salience Map (*Grad-CAM Audit*)
* Periksa visualisasi Grad-CAM pada Cell 23 Notebook 4:
  - Apakah *heatmap* atensi merah fokus tepat pada **dua baris ujung bulu sayap**?
  - Atau apakah *heatmap* justru merah di area **jari tangan manusia / meja**?
  - Jika atensi berada di jari tangan, terbukti model mengalami *shortcut learning*.

### 3. Pipeline Dua Tahap (*Two-Stage Pipeline* untuk Lapangan)
* Jika di masa depan ingin memproses foto DOC utuh dari kamera HP atau kamera Grove langsung:
  - **Tahap 1 (YOLO / Object Detection)**: Deteksi kotak pembatas (*Bounding Box*) tepat pada area sayap DOC.
  - **Tahap 2 (MobileNetV2 Classifier)**: Klasifikasi sayap yang telah dipotong otomatis oleh detektor ke dalam kelas Jantan/Betina.

---

## 5. Draf Narasi untuk Laporan Resmi / Skripsi / Publikasi

Berikut adalah contoh rumusan kalimat ilmiah yang objektif dan elegan untuk dicantumkan dalam laporan Anda:

> *"Meskipun hasil evaluasi kuantitatif awal pada dataset augmentasi Notebook 4 menunjukkan metrik akurasi 100.00% dan F1-Score 100.00%, investigasi metodologis mengungkap adanya keterbatasan representasi subjek (31 citra mentah unik). Pembagian data multi-crop tanpa partisi level-subjek (Subject-Level Partitioning) menyebabkan variasi dari subjek yang sama terdistribusi di seluruh subset (Train, Validation, dan Test), sehingga metrik 100% merefleksikan pengenalan subjek secara identik (in-distribution memorization) dan belum mencerminkan kapabilitas generalisasi penuh terhadap subjek baru di lapangan (out-of-distribution)."*

---

*Laporan ini disusun secara objektif berdasarkan audit langsung pada dataset `PP2/data/doc` dan kode pelatihan `PP2/notebook/4_training_dan_evaluasi_mobilenetv2_doc.ipynb`.*
