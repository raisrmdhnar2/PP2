# Antarmuka Modern Web Dashboard — PP2 DOC Sexing AI

Aplikasi antarmuka pengguna berbasis web modern untuk menguji dan memonitor model **DOC Sexing (MobileNetV2 INT8 on Arm Ethos-U55 NPU)** pada modul **Seeed Grove Vision AI V2**.

---

## Fitur Utama

1. **Pilihan Dual-Mode Input**:
   - 📷 **Mode Kamera Live**: Menampilkan live feed webcam laptop/PC atau kamera USB eksternal via HTML5 WebRTC. Mendukung tombol *Capture & Infer* seketika serta fitur *Auto-Infer* berkala (setiap 2 detik).
   - 🖼️ **Mode Input Gambar**: Drag & Drop atau Browse file gambar (JPG/PNG), serta tombol *Preset Quick Test Samples* (sampel uji Female #1-#3, Male #1-#3, dan Blank Frame dari dataset).
2. **Preprocessing Otomatis di PC**:
   - Citra secara otomatis di-center-crop dan di-resize ke resolusi input tensor **$160 \times 160$ RGB uint8** (76.800 Byte).
   - Menghitung CRC-16 checksum untuk menjamin integritas data serial.
3. **Komunikasi Hardware Berkecepatan Tinggi**:
   - Menggunakan protokol serial **`PP2IR`** pada baudrate **921600**.
   - Menampilkan hasil respon **`PP2_JSON:`** dari chip Himax HX6538 (Cortex-M55 + Ethos-U55).
4. **Telemetri & Metrik Lengkap**:
   - Badge Hasil Klasifikasi: **BETINA (FEMALE)** / **JANTAN (MALE)** / **BLANK / AMBIGU**.
   - Persentase Confidence & Animated Progress Bar.
   - Waktu Inferensi NPU Ethos-U55 (~3 ms).
   - Status Pengiriman Paket I2C 5-Byte ke Arduino Uno R3 / ESP32 (Slave 0x55).
   - Pratinjau Citra $160 \times 160$ persis seperti yang diterima oleh NPU.
   - Terminal Log Hardware serial interaktif.

---

## Cara Menjalankan

1. Pastikan modul Grove Vision AI V2 terhubung via kabel USB-C ke laptop/PC.
2. Jalankan server dashboard:
   ```bash
   python PP2/ui/app.py
   # Atau jika ingin port custom:
   python PP2/ui/app.py 8080
   ```
3. Buka browser Anda:
   👉 **http://localhost:8080** atau **http://127.0.0.1:8080**
4. Pilih port hardware (biasanya `/dev/ttyACM0`), lalu pilih mode **Kamera Live** atau **Input Gambar**.

