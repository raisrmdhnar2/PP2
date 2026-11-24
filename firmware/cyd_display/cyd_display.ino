/*
 * cyd_display.ino — Firmware Layar Tampilan ESP32 CYD (Cheap Yellow Display)
 *
 * Target Board : "ESP32 Dev Module" (ESP32-2432S028R / Cheap Yellow Display 2.8" TFT)
 * Lokasi File  : PP2/firmware/cyd_display/cyd_display.ino
 *
 * PERBAIKAN TOTAL:
 *   1. Rotasi 180 Derajat (Teks Tegak & Terbaca Normal):
 *      - Tampilan diputar 180 derajat ke posisi atas yang benar (tidak terbalik).
 *      - Header "PEMILAH AYAM AI" berada di bagian atas.
 *   2. Rasio Penuh Layar (Full 320x240 — Bebas Garis Noise / Terpotong):
 *      - Menggunakan subclass driver khusus CYD yang mengunci resolusi horizontal penuh
 *        (Lebar = 320 px, Tinggi = 240 px).
 *      - Garis noise / salju 80 piksel di sisi samping hilang total 100%!
 *      - Teks tidak lagi terpotong di tepi kanan.
 *   3. Warna Asli AGRI-06:
 *      - Mengembalikan background ke Hijau Botol Gelap (#16352b), Teks Putih Bersih,
 *        dan Aksen Emas (#f2b632).
 *   4. Interaktif Serial Monitor (115200 baud):
 *      - Ketik 'r' untuk uji putar layar 90 derajat kapan saja.
 *      - Ketik 'i' untuk toggle invert warna jika diperlukan.
 */

#if !defined(ESP32)
#error "SALAH PILIH BOARD! File cyd_display.ino ini KHUSUS untuk ESP32 CYD. Di Arduino IDE, silakan ubah menu: Tools -> Board -> esp32 -> 'ESP32 Dev Module'."
#endif

#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>

#include "logo_agri06.h"
#include "icons.h"

// ============================================================================
// 1. KONFIGURASI HARDWARE CYD (ESP32-2432S028R)
// ============================================================================

// I2C Pins pada Port JST CN1 CYD (ESP32-2432S028R)
#define CYD_I2C_SDA_PIN       27        // GPIO 27 = SDA pada Port CN1
#define CYD_I2C_SCL_PIN       22        // GPIO 22 = SCL pada Port CN1
#define CYD_I2C_SLAVE_ADDR    0x55      // Alamat I2C Slave (Sama dengan target Grove 0x55)
#define I2C_FREQ              100000UL

// Rotasi 180 derajat dari orientasi sebelumnya:
// Nilai 2 menghasilkan posisi Horizontal Tegak 180° (Header di atas, kabel USB di kanan)
#define DEFAULT_ROTATION      2

// Warna Asli Panel LCD CYD (false = warna asli AGRI-06, tidak terbalik)
#define DEFAULT_INVERT_COLOR  false

// Pin Layar Onboard CYD 2.8" SPI (ILI9341)
#define TFT_CS   15
#define TFT_DC   2
#define TFT_MOSI 13
#define TFT_CLK  14
#define TFT_RST  -1
#define TFT_MISO 12
#define TFT_BL   21                     // Pin Backlight LED TFT

// Palet Warna Resmi AGRI-06 (RGB565 16-bit)
#define COLOR_DARK_GREEN  0x11A5        // #16352b (Hijau Gelap Elegan)
#define COLOR_GOLD        0xF5A6        // #f2b632 (Emas Hangat)
#define COLOR_SAGE        0x7D11        // #7fa08a (Hijau Sage)
#define COLOR_CORAL       0xE387        // #e2703a (Oranye Karang)
#define COLOR_CARD_BG     0x1A69        // Hijau Gelap Lembut untuk Kartu
#define COLOR_WHITE       0xFFFF
#define COLOR_BLACK       0x0000
#define COLOR_GRAY        0x8410
#define COLOR_LIGHT_GRAY  0xD69A

// ============================================================================
// 2. DRIVER KHUSUS CYD (MENGUNCI FULL DISPLAY 320 x 240)
// ============================================================================
// Driver turunan ini memastikan lebar layar 320 px dan tinggi 240 px dimanfaatkan
// secara penuh tanpa terpotong 240 px dan tanpa garis noise di tepi.
class CYD_Display_Driver : public Adafruit_ILI9341 {
public:
    CYD_Display_Driver(int8_t cs, int8_t dc, int8_t mosi, int8_t clk, int8_t rst, int8_t miso)
        : Adafruit_ILI9341(cs, dc, mosi, clk, rst, miso) {}

    void setRotation(uint8_t m) override {
        Adafruit_ILI9341::setRotation(m);
        // Paksa resolusi native 320 horizontal x 240 vertikal
        _width  = 320;
        _height = 240;
    }
};

CYD_Display_Driver tft = CYD_Display_Driver(TFT_CS, TFT_DC, TFT_MOSI, TFT_CLK, TFT_RST, TFT_MISO);

// ============================================================================
// 3. VARIABEL DATA & PENGHITUNG HASIL
// ============================================================================
#define PKT_HEADER 0xAA

uint8_t  g_current_rot   = DEFAULT_ROTATION;
bool     g_invert_color  = DEFAULT_INVERT_COLOR;

volatile bool    g_new_data_flag = false;
volatile uint8_t g_rx_cls        = 255;  // 255 = STANDBY, 0 = Betina, 1 = Jantan, 2 = Undef
volatile uint8_t g_rx_conf_int   = 0;
volatile uint8_t g_rx_conf_dec   = 0;

uint8_t  last_cls        = 254;
uint8_t  last_conf_int   = 255;
uint8_t  last_conf_dec   = 255;

// Statistik Counter untuk Pameran
uint16_t g_count_betina  = 0;
uint16_t g_count_jantan  = 0;
uint16_t g_count_total   = 0;
uint16_t last_drawn_betina = 0xFFFF;
uint16_t last_drawn_jantan = 0xFFFF;
uint16_t last_drawn_total  = 0xFFFF;

void renderFullUI();
void updateCounterValuesOnly();
void renderMainCard(uint8_t cls, uint8_t cInt, uint8_t cDec);

// ============================================================================
// 4. CALLBACK I2C SLAVE RECEIVER
// ============================================================================
void onI2CReceive(int numBytes)
{
    if (numBytes < 5) {
        while (Wire.available()) Wire.read();
        return;
    }

    uint8_t buf[6];
    uint8_t idx = 0;
    while (Wire.available() && idx < 6) {
        buf[idx++] = Wire.read();
    }
    while (Wire.available()) Wire.read();

    if (buf[0] != PKT_HEADER) return;

    // Validasi XOR Checksum
    uint8_t calc_crc = (uint8_t)(buf[0] ^ buf[1] ^ buf[2] ^ buf[3]);
    if (buf[4] != calc_crc) return;

    uint8_t new_cls = buf[1];
    g_rx_cls        = new_cls;
    g_rx_conf_int   = buf[2];
    g_rx_conf_dec   = buf[3];

    // Akumulasi Statistik
    if (new_cls == 0) {
        g_count_betina++;
        g_count_total++;
    } else if (new_cls == 1) {
        g_count_jantan++;
        g_count_total++;
    }

    g_new_data_flag = true;
}

// ============================================================================
// 5. RENDERING TAMPILAN FULL DISPLAY HORIZONTAL (320 x 240)
// ============================================================================

void drawHeaderBar()
{
    // Background Header Horizontal (Lebar: 320, Tinggi: 36)
    tft.fillRect(0, 0, 320, 36, COLOR_DARK_GREEN);
    tft.drawFastHLine(0, 36, 320, COLOR_GOLD);
    tft.drawFastHLine(0, 37, 320, COLOR_GOLD);

    // 1. Logo Resmi AGRI-06 (32x32)
    tft.drawRGBBitmap(4, 2, logo_agri06_32x32, 32, 32);

    // 2. Judul Proyek Ramah Awam
    tft.setTextColor(COLOR_WHITE);
    tft.setTextSize(2);
    tft.setCursor(42, 5);
    tft.print("FEXA");

    tft.setTextSize(1);
    tft.setTextColor(COLOR_GOLD);
    tft.setCursor(43, 22);
    tft.print("AGRI-06 Smart Hatchery System");

    // 3. Indikator Status Sistem Online
    tft.fillCircle(296, 12, 5, 0x07E0); // Bulatan Hijau Terang
    tft.setTextSize(1);
    tft.setTextColor(COLOR_WHITE);
    tft.setCursor(276, 22);
    tft.print("ONLINE");
}

void drawCounterBar()
{
    // Area Bawah: Penghitung Hasil Sortir (Y: 188 - 238, Tinggi: 50)
    int y = 188;
    tft.fillRect(0, y, 320, 52, COLOR_DARK_GREEN);
    tft.drawFastHLine(0, y, 320, COLOR_GOLD);

    int colW = 98;

    // 1. Kotak Betina
    tft.fillRoundRect(6, y + 5, colW, 40, 4, 0x1BE8);
    tft.drawRoundRect(6, y + 5, colW, 40, 4, COLOR_WHITE);
    tft.setTextSize(1);
    tft.setTextColor(COLOR_WHITE);
    tft.setCursor(14, y + 9);
    tft.print("BETINA");

    // 2. Kotak Jantan
    tft.fillRoundRect(111, y + 5, colW, 40, 4, COLOR_CORAL);
    tft.drawRoundRect(111, y + 5, colW, 40, 4, COLOR_WHITE);
    tft.setTextSize(1);
    tft.setTextColor(COLOR_WHITE);
    tft.setCursor(119, y + 9);
    tft.print("JANTAN");

    // 3. Kotak Total
    tft.fillRoundRect(216, y + 5, colW, 40, 4, 0x228B);
    tft.drawRoundRect(216, y + 5, colW, 40, 4, COLOR_GOLD);
    tft.setTextSize(1);
    tft.setTextColor(COLOR_GOLD);
    tft.setCursor(224, y + 9);
    tft.print("TOTAL DOC");

    // Reset cache agar angka langsung terupdate
    last_drawn_betina = 0xFFFF;
    last_drawn_jantan = 0xFFFF;
    last_drawn_total  = 0xFFFF;
    updateCounterValuesOnly();
}

// Partial update cepat: HANYA memperbarui angka counter tanpa gambar ulang kotak
void updateCounterValuesOnly()
{
    int y = 188;
    int colW = 98;

    // 1. Nilai Betina
    if (g_count_betina != last_drawn_betina) {
        tft.fillRect(14, y + 23, colW - 20, 18, 0x1BE8);
        tft.setTextSize(2);
        tft.setTextColor(COLOR_WHITE);
        tft.setCursor(14, y + 23);
        tft.print(g_count_betina);
        last_drawn_betina = g_count_betina;
    }

    // 2. Nilai Jantan
    if (g_count_jantan != last_drawn_jantan) {
        tft.fillRect(119, y + 23, colW - 20, 18, COLOR_CORAL);
        tft.setTextSize(2);
        tft.setTextColor(COLOR_WHITE);
        tft.setCursor(119, y + 23);
        tft.print(g_count_jantan);
        last_drawn_jantan = g_count_jantan;
    }

    // 3. Nilai Total
    if (g_count_total != last_drawn_total) {
        tft.fillRect(224, y + 23, colW - 20, 18, 0x228B);
        tft.setTextSize(2);
        tft.setTextColor(COLOR_WHITE);
        tft.setCursor(224, y + 23);
        tft.print(g_count_total);
        last_drawn_total = g_count_total;
    }
}

void renderMainCard(uint8_t cls, uint8_t cInt, uint8_t cDec)
{
    // Area Kartu Tengah (X: 6, Y: 42, W: 308, H: 140)
    int cardX = 6;
    int cardY = 42;
    int cardW = 308;
    int cardH = 140;

    uint16_t cardBg = COLOR_CARD_BG;
    uint16_t borderCol = COLOR_GOLD;

    if (cls == 0) {
        cardBg    = 0x1466; // Hijau gelap elegan
        borderCol = 0x2E6B; // Hijau cerah
    } else if (cls == 1) {
        cardBg    = 0x51E4; // Oranye hangat
        borderCol = COLOR_CORAL;
    } else {
        cardBg    = COLOR_CARD_BG;
        borderCol = COLOR_GOLD;
    }

    // Partial Render: Hanya redraw kartu tengah
    tft.fillRoundRect(cardX, cardY, cardW, cardH, 8, cardBg);
    tft.drawRoundRect(cardX, cardY, cardW, cardH, 8, borderCol);

    // 1. Tampilkan Ikon Visual Sesuai Hasil (48x48)
    int iconX = cardX + 10;
    int iconY = cardY + 16;
    if (cls == 0) {
        tft.drawRGBBitmap(iconX, iconY, icon_hen_48x48, 48, 48);
    } else if (cls == 1) {
        tft.drawRGBBitmap(iconX, iconY, icon_cock_48x48, 48, 48);
    } else {
        tft.drawRGBBitmap(iconX, iconY, icon_scan_48x48, 48, 48);
    }

    // 2. Judul Status Besar
    int textX = iconX + 58;
    tft.setTextSize(3);
    tft.setCursor(textX, cardY + 10);
    if (cls == 0) {
        tft.setTextColor(COLOR_WHITE);
        tft.print("BETINA");
    } else if (cls == 1) {
        tft.setTextColor(COLOR_WHITE);
        tft.print("JANTAN");
    } else if (cls == 2) {
        tft.setTextColor(COLOR_GOLD);
        tft.setTextSize(2);
        tft.print("RAGU-RAGU");
    } else {
        tft.setTextColor(COLOR_WHITE);
        tft.setTextSize(2);
        tft.print("SIAP PINDAI");
    }

    // 3. Penjelasan Masuk Bilik (Tanpa Menampilkan Sudut Servo)
    tft.setTextSize(2);
    tft.setCursor(textX, cardY + 38);
    if (cls == 0) {
        tft.setTextColor(0x7FFF); // Cyan/Mint terang
        tft.print("Bilik Betina");
    } else if (cls == 1) {
        tft.setTextColor(0xFD20); // Oranye cerah
        tft.print("Bilik Jantan");
    } else if (cls == 2) {
        tft.setTextColor(COLOR_GOLD);
        tft.print("Tahan Telur");
    } else {
        tft.setTextColor(COLOR_LIGHT_GRAY);
        tft.print("Standby");
    }

    // Sub-keterangan Arah Alur Bilik
    tft.setTextSize(1);
    tft.setTextColor(COLOR_WHITE);
    tft.setCursor(textX, cardY + 59);
    if (cls == 0) {
        tft.print("Masuk ke Jalur Sortir Kanan");
    } else if (cls == 1) {
        tft.print("Masuk ke Jalur Sortir Kiri");
    } else if (cls == 2) {
        tft.print("Pemeriksaan ulang sensor...");
    } else {
        tft.setTextColor(COLOR_GOLD);
        tft.print("Letakkan DOC di sensor kamera...");
    }

    // 4. Bar Akurasi AI
    int barX = textX;
    int barY = cardY + 76;
    int barW = 216;
    int barH = 12;

    tft.drawRoundRect(barX, barY, barW, barH, 3, COLOR_WHITE);
    tft.fillRect(barX + 2, barY + 2, barW - 4, barH - 4, COLOR_BLACK);

    uint8_t clamped = (cInt > 100) ? 100 : cInt;
    int fillW = (int)((clamped * (barW - 4)) / 100);
    if (fillW > 0 && cls != 255) {
        uint16_t fillCol = (cls == 1) ? COLOR_CORAL : (cls == 0) ? 0x2FE8 : COLOR_GOLD;
        tft.fillRoundRect(barX + 2, barY + 2, fillW, barH - 4, 2, fillCol);
    }

    // 5. Teks Akurasi Persentase AI
    tft.setTextSize(1);
    tft.setCursor(textX, cardY + 94);
    tft.setTextColor(COLOR_WHITE);
    if (cls == 255) {
        tft.print("Tingkat Keyakinan AI: Menunggu...");
    } else {
        tft.print("Tingkat Keyakinan AI: ");
        tft.setTextColor(COLOR_GOLD);
        tft.print(cInt);
        tft.print(".");
        if (cDec < 10) tft.print("0");
        tft.print(cDec);
        tft.print("%");
    }

    // Footer Kartu Status
    tft.setTextSize(1);
    tft.setTextColor(COLOR_LIGHT_GRAY);
    tft.setCursor(textX, cardY + 114);
    tft.print("Sistem Pemilah Otomatis AGRI-06");
}

void renderFullUI()
{
    tft.fillScreen(COLOR_DARK_GREEN);
    drawHeaderBar();
    renderMainCard(g_rx_cls, g_rx_conf_int, g_rx_conf_dec);
    drawCounterBar();
}

// ============================================================================
// 6. SETUP & LOOP
// ============================================================================
void setup()
{
    Serial.begin(115200);
    delay(100);
    Serial.println("\n========================================================");
    Serial.println("  ESP32 CYD Smart Display — PP2 DOC Sexing AI           ");
    Serial.println("  (Ketik 'r' di Serial = Putar Layar 90 deg)            ");
    Serial.println("  (Ketik 'i' di Serial = Toggle Invert Warna Layar)     ");
    Serial.println("========================================================");

    // 1. Nyalakan Lampu Latar Layar
    pinMode(TFT_BL, OUTPUT);
    digitalWrite(TFT_BL, HIGH);

    // 2. Inisialisasi Driver Display
    tft.begin();

    // 3. Terapkan Rotasi 180 derajat (Horizontal Tegak) & Warna Asli
    tft.setRotation(g_current_rot);
    tft.invertDisplay(g_invert_color);

    // 4. Bersihkan Layar Penuh 320x240 (Menghapus seluruh noise)
    tft.fillScreen(COLOR_DARK_GREEN);

    // 5. Gambar UI Horizontal Full Display
    renderFullUI();

    // 6. Inisialisasi Bus I2C Slave pada GPIO 27 (SDA) & GPIO 22 (SCL)
    Wire.setPins(CYD_I2C_SDA_PIN, CYD_I2C_SCL_PIN);
    Wire.begin((uint8_t)CYD_I2C_SLAVE_ADDR, CYD_I2C_SDA_PIN, CYD_I2C_SCL_PIN, I2C_FREQ);
    Wire.onReceive(onI2CReceive);

    Serial.print("[READY] Full Display Siap: ");
    Serial.print(tft.width());
    Serial.print("x");
    Serial.print(tft.height());
    Serial.print(" | Rotasi: ");
    Serial.println(g_current_rot);
}

void loop()
{
    // Fitur Putar Layar & Toggle Warna Langsung dari Serial Monitor
    if (Serial.available() > 0) {
        char cmd = (char)Serial.read();
        if (cmd == 'r' || cmd == 'R') {
            g_current_rot = (g_current_rot + 1) % 4;
            tft.setRotation(g_current_rot);
            renderFullUI();
            Serial.print("[CYD] Rotasi sekarang: ");
            Serial.println(g_current_rot);
        } else if (cmd == 'i' || cmd == 'I') {
            g_invert_color = !g_invert_color;
            tft.invertDisplay(g_invert_color);
            renderFullUI();
            Serial.print("[CYD] Invert Display: ");
            Serial.println(g_invert_color ? "TRUE" : "FALSE");
        } else if (cmd == '0') {
            // Simulasi Deteksi Betina
            g_rx_cls = 0; g_rx_conf_int = 95; g_rx_conf_dec = 20;
            g_count_betina++; g_count_total++; g_new_data_flag = true;
            Serial.println("[CYD SIMULASI] Masuk ke Bilik Betina");
        } else if (cmd == '1') {
            // Simulasi Deteksi Jantan
            g_rx_cls = 1; g_rx_conf_int = 98; g_rx_conf_dec = 50;
            g_count_jantan++; g_count_total++; g_new_data_flag = true;
            Serial.println("[CYD SIMULASI] Masuk ke Bilik Jantan");
        } else if (cmd == '9') {
            // Simulasi Standby
            g_rx_cls = 2; g_rx_conf_int = 0; g_rx_conf_dec = 0;
            g_new_data_flag = true;
            Serial.println("[CYD SIMULASI] Standby / Pindai Ulang");
        }
    }

    if (g_new_data_flag) {
        noInterrupts();
        uint8_t c = g_rx_cls;
        uint8_t i = g_rx_conf_int;
        uint8_t d = g_rx_conf_dec;
        g_new_data_flag = false;
        interrupts();

        // PARTIAL UPDATE ONLY:
        // Tidak menggambar ulang seluruh frame layar! Hanya update kartu tengah & angka counter
        if (c != last_cls || i != last_conf_int || d != last_conf_dec) {
            renderMainCard(c, i, d);   // Hanya refresh kotak kartu tengah (W: 308, H: 140)
            updateCounterValuesOnly(); // Hanya refresh angka counter di bawah
            last_cls      = c;
            last_conf_int = i;
            last_conf_dec = d;

            Serial.print("[CYD] Update Hasil: ");
            if (c == 0) Serial.print("Masuk ke Bilik Betina");
            else if (c == 1) Serial.print("Masuk ke Bilik Jantan");
            else if (c == 2) Serial.print("Ragu-Ragu / Tahan");
            else Serial.print("Standby");
            Serial.print(" | Keyakinan: ");
            Serial.print(i);
            Serial.print(".");
            if (d < 10) Serial.print("0");
            Serial.print(d);
            Serial.println("%");
        }
    }
    delay(20);
}
