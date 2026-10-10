# Konsep Sistem Klasifikasi Kelamin Anak Ayam (Fase PoC: Ayam Dewasa)

## 1. Deskripsi Proyek
Proyek ini bertujuan untuk membangun sistem *sorting* biner berbasis *Edge AI* untuk mengklasifikasikan jenis kelamin anak ayam (*Day Old Chick* - DOC). Karena tingkat kesulitan klasifikasi DOC yang tinggi, proyek dimulai dengan tahap *Proof of Concept* (PoC) menggunakan dataset ayam dewasa (yang memiliki dimorfisme seksual jelas) untuk menguji keandalan *hardware*, model klasifikasi, dan komunikasi antar komponen.

## 2. Spesifikasi Perangkat
*   **Modul AI & Visi:** Grove Vision AI V2 + Raspberry Pi Camera
*   **Mikrokontroler Utama:** ESP32
*   **Aktuator:** Motor Servo (180 derajat)
*   **Model Machine Learning:** MobileNetV2 (Klasifikasi Biner)

## 3. Alur Kerja Sistem (Workflow)
1.  **Input Gambar:** Objek (ayam atau gambar uji) diletakkan di depan kamera.
2.  **Inferensi AI:** Grove Vision AI V2 memproses gambar menggunakan model MobileNetV2.
3.  **Klasifikasi:** Model menghasilkan dua output biner:
    *   `1` : Jantan
    *   `0` : Betina
4.  **Komunikasi Data:** Grove Vision AI V2 mengirimkan hasil prediksi beserta tingkat probabilitas (*confidence score*) ke ESP32 melalui protokol komunikasi I2C.
5.  **Aktuasi Fisik:** ESP32 membaca data dan mengarahkan servo:
    *   `0 Derajat`   : Jalur Betina (Menerima input 0)
    *   `180 Derajat` : Jalur Jantan (Menerima input 1)
    *   `90 Derajat`  : Posisi *Standby* (Center point)

## 4. Arsitektur Teknis & Optimasi

### A. Pengolahan Model di Edge
Agar inferensi berjalan optimal di atas NPU Arm Ethos-U55 (Grove Vision AI V2), model MobileNetV2 tidak bisa dijalankan secara mentah. Model harus melalui tahap:
*   Kuantisasi ke **INT8** via TensorFlow Lite.
*   Kompilasi menggunakan **Arm Vela Compiler**.

### B. Komunikasi I2C (Inter-Integrated Circuit)
*   **Topologi:** ESP32 bertindak sebagai **I2C Master**, sedangkan Grove Vision AI V2 bertindak sebagai **I2C Slave** (umumnya di *address* `0x62`).
*   **Metode:** ESP32 melakukan *polling* secara berkala untuk mengambil data hasil inferensi terbaru dari Grove Vision.

### C. Logika Pergerakan Servo (State Machine)
Untuk mencegah program *hang* atau menunda pembacaan sensor berikutnya, ESP32 harus menghindari penggunaan *blocking delay* (`delay()`). 
*   **Logika:** Servo di 90° -> Data masuk -> Gerak ke 0°/180° -> Jeda waktu *non-blocking* (menggunakan fungsi `millis()`) -> Kembali ke 90°.

## 5. Penanganan Kasus Uji & Skenario Khusus

### A. Penanganan "Frame Kosong" (Tidak Ada Objek)
Jika kamera menyorot ruang kosong, model biner akan memaksakan prediksi yang tidak relevan dengan *confidence* yang fluktuatif. Solusi yang diimplementasikan:
1.  **Logic Threshold (Software ESP32):** ESP32 hanya akan merespons (menggerakkan servo) jika nilai *confidence* dari Grove Vision AI di atas batas minimum (misalnya > 80%). Jika di bawah itu, servo tetap di 90°.
2.  **Kelas Background (Sisi Model):** Menambahkan kelas ke-3 (`kelas 2: kosong / background`) saat pelatihan model agar sistem tahu kapan harus diam.

### B. Metode Pengujian Gambar
1.  **Pengujian Fisik:** Menggunakan kertas yang di-print. Perlu diperhatikan penggunaan kertas *matte/doff* (tidak mengkilap) dan pencahayaan merata untuk menghindari *glare* yang merusak akurasi kamera.
2.  **Pengujian Langsung via Laptop:** Menggunakan platform web **SenseCraft AI Studio** dari Seeed Studio. Grove Vision AI V2 disambungkan ke laptop via USB-C, dan gambar pengujian bisa langsung diunggah dari laptop tanpa perlu dicetak, mempercepat proses *testing* dataset.

## 6. Catatan Transisi ke Fase DOC (Anak Ayam)
Ketika sistem berpindah dari ayam dewasa ke DOC, beberapa penyesuaian keras (*hardware*) perlu dilakukan:
*   Pencahayaan harus optimal (disarankan menggunakan *LED Ring*).
*   Fokus makro pada kamera, mengingat klasifikasi DOC (*feather sexing* atau *vent sexing*) bergantung pada detail fisik yang sangat kecil.