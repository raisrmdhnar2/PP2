# Rekap Lengkap Wiring & Integrasi Hardware (PP2 DOC Sexing AI)

Dokumen ini adalah panduan referensi wiring dan topologi koneksi kabel antara 3 perangkat utama:
1. **Seeed Grove Vision AI Module V2** (Vision AI Sensor / NPU)
2. **Arduino Uno R3** (Controller Utama & Penggerak Motor Servo Metal Gear)
3. **ESP32 CYD (Cheap Yellow Display / ESP32-2432S028R)** (Unit Layar Tampilan TFT Warna 2.8")
4. **Motor Servo MG** (Aktuator Pemilah / Metal Gear)

---

## 1. Diagram Topologi Sistem

```mermaid
flowchart TD
    subgraph DAYA ["Sistem Catu Daya (Power)"]
        USB_PC["USB PC / Laptop (5V)"]
        EXT_5V["Adaptor Eksternal 5V (2A - 3A)"]
    end

    subgraph GROVE ["1. Seeed Grove Vision AI V2 (3.3V)"]
        G_SCL["PA2 (I2C SCL)"]
        G_SDA["PA3 (I2C SDA)"]
        G_GND["GND"]
    end

    subgraph UNO ["2. Arduino Uno R3 (Controller)"]
        U_A4["Pin A4 (SDA)"]
        U_A5["Pin A5 (SCL)"]
        U_D9["Pin 9 (PWM Servo)"]
        U_GND["GND"]
        U_5V["5V"]
    end

    subgraph CYD ["3. ESP32 CYD (Layar TFT Warna)"]
        C_SCL["GPIO 22 (SCL)"]
        C_SDA["GPIO 27 (SDA)"]
        C_GND["GND"]
        C_PWR["Port USB / Pin VIN (5V)"]
    end

    subgraph SERVO ["4. Motor Servo MG (Metal Gear)"]
        S_SIG["Signal (Kuning/Oranye)"]
        S_VCC["VCC (Merah)"]
        S_GND["GND (Cokelat/Hitam)"]
    end

    %% Koneksi Daya
    USB_PC -->|"USB-C 1"| GROVE
    USB_PC -->|"USB-B 2"| UNO
    USB_PC -->|"USB-C / Micro 3"| CYD
    EXT_5V -->|5V Mandiri| S_VCC

    %% Common Ground
    G_GND <===> U_GND
    U_GND <===> C_GND
    U_GND <===> S_GND
    EXT_5V -.->|GND Bersama| U_GND

    %% Jalur Sinyal I2C Bersama (Shared I2C Bus)
    G_SDA <---> U_A4
    U_A4 <---> C_SDA
    G_SCL <---> U_A5
    U_A5 <---> C_SCL

    %% Jalur Sinyal Servo
    U_D9 ---> S_SIG
```

---

## 2. Tabel Detail Koneksi Pin-ke-Pin

### A. Jalur Komunikasi Bus I2C (SCL & SDA)
Jalur I2C dihubungkan secara paralel (*Shared I2C Bus*):

| Dari Pin Perangkat | Warna Kabel Rekomendasi | Terhubung Ke Pin | Fungsi |
| :--- | :---: | :--- | :--- |
| **Grove Vision AI V2** (Pin `PA3`) | **Putih** | **Arduino Uno `Pin A4`** & **ESP32 CYD `GPIO 27`** | Data I2C (SDA) |
| **Grove Vision AI V2** (Pin `PA2`) | **Kuning** | **Arduino Uno `Pin A5`** & **ESP32 CYD `GPIO 22`** | Clock I2C (SCL) |
| **Grove Vision AI V2** (`GND`) | **Hitam** | **Arduino Uno `GND`** & **ESP32 CYD `GND`** | Ground Bersama (*Common GND*) |

> [!IMPORTANT]
> **Tegangan Logika I2C (3.3V Safe):**
> Grove Vision AI V2 dan ESP32 CYD bekerja pada logika **3.3V**, sedangkan Arduino Uno bekerja pada 5V.
> Pada kode Arduino Uno (`mainv3.ino`), *internal pull-up* 5V pada pin A4 & A5 **dimatikan secara software** (`digitalWrite(SDA, LOW); digitalWrite(SCL, LOW);`) sehingga bus I2C murni ditarik ke 3.3V oleh ESP32 CYD / Grove. Hal ini **100% aman** dan tidak merusak chip Grove Vision AI V2.

---

### B. Jalur Penggerak Motor Servo MG (Metal Gear)

| Pin Servo MG | Warna Kabel | Hubungkan Ke | Catatan Penting |
| :--- | :---: | :--- | :--- |
| **Signal (PWM)** | **Oranye / Kuning** | **Pin 9 Arduino Uno (`D9`)** | Sinyal kendali 50 Hz Hardware Timer1 Uno |
| **VCC (Power)** | **Merah** | **Kutub Positif (+) Adaptor 5V Eksternal** | **DILARANG** colok ke pin 3.3V/5V CYD/Uno karena servo MG menarik lonjakan arus 1.5A–2.5A |
| **GND (Ground)** | **Coklat / Hitam** | **GND Bersama (Kutub Negatif Adaptor & GND Uno)** | Wajib disatukan agar referensi ground sama |

---

### C. Pinout Port Konektor ESP32 CYD (Cheap Yellow Display)
ESP32 CYD (ESP32-2432S028R) memiliki port konektor eksternal JST 4-pin (biasanya berlabel **CN1** atau **P3**) di dekat slot SD Card / tombol boot:

| Pin Port CYD | Label Pin CYD | Hubungkan Ke | Keterangan |
| :---: | :---: | :--- | :--- |
| **Pin 1** | `GND` | **GND Arduino Uno & GND Grove** | Ground bersama |
| **Pin 2** | `3V3` atau `5V` *(NC / Kosong)* | *Tidak wajib disambung jika CYD dicolok USB* | CYD diberi daya via kabel USB-nya sendiri |
| **Pin 3** | **`IO27` (SDA)** | **Arduino Uno `Pin A4` & Grove `PA3`** | Jalur I2C Data |
| **Pin 4** | **`IO22` (SCL)** | **Arduino Uno `Pin A5` & Grove `PA2`** | Jalur I2C Clock |

---

## 3. Alur Kerja Sistem (Data Flow)

1. **Grove Vision AI V2**:
   - Menerima citra bulu sayap DOC (kamera atau input gambar UI).
   - Menjalankan inferensi model MobileNetV2 INT8 pada Ethos-U55 NPU.
   - Mengambil keputusan berdasarkan threshold optimal **38.0%**:
     - $P(\text{Jantan}) \ge 38.0\% \rightarrow \text{Class 1 (Jantan)}$
     - $P(\text{Jantan}) < 38.0\% \rightarrow \text{Class 0 (Betina)}$
   - Mengirim paket protokol 5-byte via I2C:
     `[0xAA, class_id, conf_int, conf_dec, checksum]`
2. **Arduino Uno R3**:
   - Menerima paket 5-byte di Slave Address `0x55`.
   - Menggerakkan tuas motor servo MG:
     - **0°**: Jalur Betina (Kiri)
     - **180°**: Jalur Jantan (Kanan)
     - **90°**: Standby / Center
   - Mengatur jeda *cooldown* 4 detik sebelum kembali ke posisi awal 90°.
3. **ESP32 CYD**:
   - Menerima hasil deteksi di bus I2C.
   - Menampilkan visual antarmuka pameran di layar LCD warna 2.8":
     - Logo / Judul Proyek (PP2 DOC Sexing AI)
     - Status: `WAITING...` / `BETINA (HEN)` (Hijau) / `JANTAN (COCK)` (Kuning/Oranye)
     - Nilai Kepastian (*Confidence*) dalam persentase & Bar Grafis
     - Posisi Sudut Servo saat ini
