/*
 * mainv2.ino — Firmware Master Controller Arduino Uno R3 (PP2 Egg Sexing AI)
 *
 * Lokasi File : PP2/firmware/mainv2/mainv2.ino
 * Target Board: "Arduino Uno" (ATmega328P @ 16 MHz, 2 KB SRAM, 32 KB Flash)
 *
 * Perbaikan Stabilitas Multi-Deteksi & Auto-Reset:
 *   1. Mengaktifkan Hardware I2C Timeout (Wire.setWireTimeout(25000, true)) agar
 *      bus I2C bersama (A4 SDA & A5 SCL) TIDAK PERNAH hang/nyangkut saat terjadi
 *      lonjakan arus motor servo atau pergantian Master/Slave dengan Grove Vision AI V2.
 *   2. Memisahkan waktu transaksi I2C OLED (display.display()) dari waktu gerak awal
 *      motor servo (inrush current spike), serta meniadakan spam refresh OLED setiap 1 detik
 *      selama masa tahan (cooldown) agar bus I2C tidak pernah macet di tengah jalan.
 *   3. Siklus Otomatis Penuh untuk Setiap Gambar (Berulang Tanpa Batas):
 *        [State Awal]   Servo di 90° (Center) & OLED "STATUS: WAITING AI..."
 *             |
 *             v  (Menerima hasil deteksi dari Grove Vision AI V2 via I2C 0x55)
 *        [Aktuasi]      OLED tampilkan Hasil (Betina/Jantan + Confidence)
 *                       & Servo bergerak ke 0° (Betina) atau 180° (Jantan)
 *             |
 *             v  (Tahan selama Cooldown 4 detik)
 *        [Auto-Reset]   Servo kembali ke Center Point (90°) -> tunggu gerak selesai ->
 *                       OLED kembali ke State Awal ("STATUS: WAITING AI...") ->
 *                       Siap menerima deteksi berikutnya (#002, #003, dst.).
 */

#include <Arduino.h>
#include <Wire.h>
#include <Servo.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <avr/io.h>

// ============================================================================
// 1. KONFIGURASI PIN & KONSTANTA PERANGKAT KERAS (ARDUINO UNO R3)
// ============================================================================

// Alamat I2C Slave Arduino Uno R3 (penerima paket 5-Byte dari Grove Vision AI V2)
#define GROVE_I2C_SLAVE_ADDR  0x55
#define I2C_BUS_FREQ          100000UL

// Konfigurasi Layar OLED SSD1306 128x64 pada Pin A4 (SDA) & A5 (SCL)
#define OLED_I2C_ADDR         0x3C
#define SCREEN_WIDTH          128
#define SCREEN_HEIGHT         64
#define OLED_RESET            -1

// Konfigurasi Motor Servo 3-Pin pada Pin 9 (Hardware Timer1 16-bit ATmega328P)
#define SERVO_PIN             9
#define SERVO_MIN_US          600     // Batas pulsa aman 0 derajat (us) agar gir tidak mentok
#define SERVO_MAX_US          2300    // Batas pulsa aman 180 derajat (us) agar gir tidak mentok

// Keputusan Sudut Servo yang Dikomandoi oleh Arduino Uno R3
#define SERVO_ANGLE_BETINA    0       // 0 derajat   -> Jalur Sortir Kiri (Betina)
#define SERVO_ANGLE_STANDBY   90      // 90 derajat  -> Center Point / Default Standby
#define SERVO_ANGLE_JANTAN    180     // 180 derajat -> Jalur Sortir Kanan (Jantan)

// Durasi Tahan (Cooldown) Sebelum Servo Kembali ke Center (90°) & OLED Kembali ke WAITING AI
#define SORT_COOLDOWN_MS      4000UL  // 4 detik tahan posisi sortir lalu reset ke 90° & WAITING AI
#define UNDEF_COOLDOWN_MS     2500UL  // 2,5 detik tampil Undefined lalu reset ke WAITING AI
#define ERROR_COOLDOWN_MS     2000UL  // 2 detik auto-recovery dari error paket I2C

// Protokol Paket 5-Byte dari Grove Vision AI V2
#define PP2_PKT_HEADER        0xAA
#define PP2_PKT_SIZE          5

// ============================================================================
// 2. DEFINISI STATE MACHINE ARDUINO UNO R3 (OLED & SERVO)
// ============================================================================
typedef enum {
    STATE_BOOT_INIT = 0,
    STATE_WAITING_DATA,         // State Awal (Default): Servo 90° (Center), OLED menunggu input AI
    STATE_SORT_BETINA,          // Deteksi Betina: Servo 0° (Kiri), lalu kembali ke 90° & WAITING AI
    STATE_SORT_JANTAN,          // Deteksi Jantan: Servo 180° (Kanan), lalu kembali ke 90° & WAITING AI
    STATE_UNDEFINED_NO_EGG,     // Undefined (Frame Kosong): Servo tetap 90°
    STATE_UNDEFINED_LOW_CONF,   // Undefined (Ragu <80%): Servo tetap 90°
    STATE_I2C_ERROR             // Error paket I2C: Servo tetap 90°
} uno_state_t;

// ============================================================================
// 3. OBJEK & VARIABEL GLOBAL (HEMAT SRAM UNTUK ATmega328P 2 KB)
// ============================================================================
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET, I2C_BUS_FREQ, I2C_BUS_FREQ);
Servo            eggServo;

volatile bool     g_new_packet_flag   = false;
volatile bool     g_packet_err_flag   = false;
volatile uint8_t  g_rx_class_id       = 2;
volatile uint8_t  g_rx_conf_int       = 0;
volatile uint8_t  g_rx_conf_dec       = 0;
volatile uint16_t g_packet_count      = 0;

uno_state_t g_current_state           = STATE_BOOT_INIT;
bool        g_oled_ready              = false;
uint8_t     g_current_angle           = SERVO_ANGLE_STANDBY;
uint8_t     g_disp_conf_int           = 0;
uint8_t     g_disp_conf_dec           = 0;
uint32_t    g_state_enter_ms          = 0;

// Deklarasi callback I2C Slave
void onGroveReceive(int numBytes);

// ============================================================================
// 4. FUNGSI PEMULIHAN REGISTER HARDWARE TWI SLAVE 0x55 (TANPA RESET WIRE)
// ============================================================================
// Sangat penting pada ATmega328P:
// Jangan memanggil Wire.begin(0x55) berulang-ulang setelah display.display() karena
// Wire.begin() mengaktifkan ulang pull-up 5V dan mereset state internal TWI.
// Cukup pastikan register TWAR = (0x55 << 1) dan bit TWEA (Enable ACK) aktif!
static void ensureTwiSlaveReady()
{
    // Jika sebelumnya terjadi timeout pada bus I2C, bersihkan flag timeout-nya
#if defined(WIRE_HAS_TIMEOUT)
    if (Wire.getWireTimeoutFlag()) {
        Wire.clearWireTimeoutFlag();
    }
#endif

    // Tunggu maksimal 2 ms jika kondisi STOP dari transaksi Master OLED masih berlangsung
    uint16_t guard = 2000U;
    while ((TWCR & _BV(TWSTO)) && (guard > 0U)) {
        delayMicroseconds(1);
        guard--;
    }

    // Matikan pull-up internal 5V pada A4 (SDA) & A5 (SCL) agar bus tetap 3.3V
    digitalWrite(SDA, LOW);
    digitalWrite(SCL, LOW);

    // Pastikan alamat Slave 0x55 dan ACK otomatis (TWEA) aktif di hardware TWI
    TWAR = (uint8_t)(GROVE_I2C_SLAVE_ADDR << 1);
    TWCR = _BV(TWEN) | _BV(TWIE) | _BV(TWEA) | _BV(TWINT);
}

// ============================================================================
// 5. FUNGSI KONTROL MOTOR SERVO 3-PIN (PIN 9 HARDWARE TIMER1)
// ============================================================================
void setServoAngle(uint8_t targetAngle)
{
    if (targetAngle > 180) targetAngle = 180;
    g_current_angle = targetAngle;
    eggServo.write(g_current_angle);

    Serial.print(F("[SERVO] Pin 9 -> Sudut: "));
    Serial.print(g_current_angle);
    Serial.println(F(" deg"));
}

// ============================================================================
// 6. INTERRUPT CALLBACK I2C SLAVE (MENERIMA 5 BYTE DARI GROVE VISION AI V2)
// ============================================================================
void onGroveReceive(int numBytes)
{
    if (numBytes < PP2_PKT_SIZE) {
        while (Wire.available()) {
            Wire.read();
        }
        return;
    }

    uint8_t pkt[PP2_PKT_SIZE];
    for (uint8_t i = 0; i < PP2_PKT_SIZE; ++i) {
        pkt[i] = (uint8_t)Wire.read();
    }
    while (Wire.available()) {
        Wire.read();
    }

    // Validasi Header (0xAA) & XOR Checksum
    uint8_t calc_crc = (uint8_t)(pkt[0] ^ pkt[1] ^ pkt[2] ^ pkt[3]);
    if (pkt[0] != PP2_PKT_HEADER || pkt[4] != calc_crc || pkt[1] > 2) {
        g_packet_err_flag = true;
        return;
    }

    // Selalu terima paket valid terbaru dari Grove Vision AI V2 agar tidak pernah nyangkut
    g_rx_class_id     = pkt[1];
    g_rx_conf_int     = pkt[2];
    g_rx_conf_dec     = pkt[3];
    g_packet_count    = g_packet_count + 1;
    g_new_packet_flag = true;
}

// ============================================================================
// 7. FUNGSI RENDERING STATE MACHINE LAYAR OLED SSD1306 (HEMAT RAM / PROGMEM)
// ============================================================================

static void drawConfidenceBar(uint8_t confInt)
{
    const int barX = 4;
    const int barY = 41;
    const int barW = 120;
    const int barH = 9;

    display.drawRect(barX, barY, barW, barH, SSD1306_WHITE);

    uint8_t clamped = (confInt > 100) ? 100 : confInt;
    int fillW = (int)(((uint16_t)clamped * (uint16_t)(barW - 4) + 50U) / 100U);
    if (fillW > 0) {
        display.fillRect(barX + 2, barY + 2, fillW, barH - 4, SSD1306_WHITE);
    }

    // Garis vertikal penanda Threshold 80% (80% dari 116 px = 93 px)
    int thX = barX + 2 + 93;
    display.drawLine(thX, barY - 2, thX, barY + barH + 1, SSD1306_INVERSE);
}

static void drawHeaderBanner(uint16_t pktCount)
{
    display.fillRect(0, 0, SCREEN_WIDTH, 11, SSD1306_WHITE);
    display.setTextSize(1);
    display.setTextColor(SSD1306_BLACK);
    display.setCursor(3, 2);
    display.print(F("PP2 EGG SEXING [#"));
    uint16_t c = pktCount % 1000U;
    if (c < 100U) display.print('0');
    if (c < 10U)  display.print('0');
    display.print(c);
    display.print(']');
    display.setTextColor(SSD1306_WHITE);
}

static void printConfidenceLine(uint8_t cInt, uint8_t cDec, const __FlashStringHelper *suffix)
{
    display.setCursor(4, 28);
    display.print(F("CONF  : "));
    display.print(cInt);
    display.print('.');
    if (cDec < 10U) display.print('0');
    display.print(cDec);
    display.print(F("%"));
    if (suffix != NULL) {
        display.print(suffix);
    }
}

void renderOledState()
{
    if (!g_oled_ready) {
        ensureTwiSlaveReady();
        return;
    }

    display.clearDisplay();

    switch (g_current_state) {
        case STATE_BOOT_INIT: {
            display.fillRect(0, 0, SCREEN_WIDTH, 12, SSD1306_WHITE);
            display.setTextSize(1);
            display.setTextColor(SSD1306_BLACK);
            display.setCursor(5, 2);
            display.print(F("PP2 ARDUINO UNO R3"));

            display.setTextColor(SSD1306_WHITE);
            display.setCursor(4, 16);
            display.print(F("SELF-TEST SERVO..."));
            display.setCursor(4, 28);
            display.print(F("I2C Slv: 0x55 (A4/A5)"));
            display.setCursor(4, 40);
            display.print(F("OLED   : 0x3C (A4/A5)"));
            display.setCursor(4, 52);
            display.print(F("Servo  : PIN 9 (5V)"));
            break;
        }

        case STATE_WAITING_DATA: {
            drawHeaderBanner(g_packet_count);
            display.setCursor(4, 15);
            display.print(F("STATUS: WAITING AI..."));
            display.setCursor(4, 28);
            display.print(F("CONF  : --.-- %"));
            drawConfidenceBar(0);
            display.setCursor(4, 54);
            display.print(F("SERVO : 90 DEG (READY)"));
            break;
        }

        case STATE_SORT_BETINA: {
            drawHeaderBanner(g_packet_count);
            display.setCursor(4, 15);
            display.print(F("HASIL : BETINA (HEN)"));
            printConfidenceLine(g_disp_conf_int, g_disp_conf_dec, NULL);
            drawConfidenceBar(g_disp_conf_int);
            display.setCursor(4, 54);
            display.print(F("SERVO : 0 DEG -> 90"));
            break;
        }

        case STATE_SORT_JANTAN: {
            drawHeaderBanner(g_packet_count);
            display.setCursor(4, 15);
            display.print(F("HASIL : JANTAN (COCK)"));
            printConfidenceLine(g_disp_conf_int, g_disp_conf_dec, NULL);
            drawConfidenceBar(g_disp_conf_int);
            display.setCursor(4, 54);
            display.print(F("SERVO : 180 DEG -> 90"));
            break;
        }

        case STATE_UNDEFINED_NO_EGG: {
            drawHeaderBanner(g_packet_count);
            display.setCursor(4, 15);
            display.print(F("HASIL : UNDEFINED"));
            display.setCursor(4, 28);
            display.print(F("CONF  : 0.00% (NO EGG)"));
            drawConfidenceBar(0);
            display.setCursor(4, 54);
            display.print(F("SERVO : 90 DEG (HOLD)"));
            break;
        }

        case STATE_UNDEFINED_LOW_CONF: {
            drawHeaderBanner(g_packet_count);
            display.setCursor(4, 15);
            display.print(F("HASIL : UNDEF (<80%)"));
            printConfidenceLine(g_disp_conf_int, g_disp_conf_dec, F(" (RAGU)"));
            drawConfidenceBar(g_disp_conf_int);
            display.setCursor(4, 54);
            display.print(F("SERVO : 90 DEG (HOLD)"));
            break;
        }

        case STATE_I2C_ERROR: {
            drawHeaderBanner(g_packet_count);
            display.setCursor(4, 18);
            display.print(F("WARN  : I2C PKT ERROR"));
            display.setCursor(4, 32);
            display.print(F("Check GND/A4/A5!"));
            display.setCursor(4, 54);
            display.print(F("SERVO : 90 DEG (SAFE)"));
            break;
        }
    }

    display.display();

    // Segera pastikan modul hardware TWI kembali siap mendengarkan Slave 0x55
    ensureTwiSlaveReady();
}

// ============================================================================
// 8. TRANSISI STATE MACHINE (PISAHKAN TRANSAKSI I2C OLED & LONJAKAN ARUS SERVO)
// ============================================================================
void transitionToState(uno_state_t nextState)
{
    g_current_state = nextState;

    if (nextState == STATE_SORT_BETINA) {
        // 1. Update OLED terlebih dahulu saat motor servo belum menyedot arus
        renderOledState();
        delay(25);
        // 2. Gerakkan Servo ke 0 derajat (Betina)
        setServoAngle(SERVO_ANGLE_BETINA);
        g_state_enter_ms = millis();
    }
    else if (nextState == STATE_SORT_JANTAN) {
        // 1. Update OLED terlebih dahulu saat motor servo belum menyedot arus
        renderOledState();
        delay(25);
        // 2. Gerakkan Servo ke 180 derajat (Jantan)
        setServoAngle(SERVO_ANGLE_JANTAN);
        g_state_enter_ms = millis();
    }
    else if (nextState == STATE_WAITING_DATA) {
        // Saat kembali ke State Awal:
        // 1. Kembalikan Servo ke Center Point (90 derajat) terlebih dahulu
        if (g_current_angle != SERVO_ANGLE_STANDBY) {
            setServoAngle(SERVO_ANGLE_STANDBY);
            // Tunggu 450 ms agar motor servo selesai berputar ke 90° & arus kembali tenang
            delay(450);
        } else {
            setServoAngle(SERVO_ANGLE_STANDBY);
        }
        // 2. Setelah servo tenang di 90°, update OLED kembali ke "STATUS: WAITING AI..."
        renderOledState();
        g_state_enter_ms = millis();
    }
    else {
        // STATE_UNDEFINED_NO_EGG / STATE_UNDEFINED_LOW_CONF / STATE_I2C_ERROR / STATE_BOOT_INIT
        setServoAngle(SERVO_ANGLE_STANDBY);
        renderOledState();
        g_state_enter_ms = millis();
    }

    ensureTwiSlaveReady();
}

// Helper pencetak log hasil klasifikasi ke Serial Monitor
static void logClassificationSerial(uint16_t pktNum, uint8_t cls, uint8_t cInt, uint8_t cDec)
{
    Serial.print(F("[#"));
    Serial.print(pktNum);
    Serial.print(F("] GROVE AI -> CLASS "));
    Serial.print(cls);
    if (cls == 0) {
        Serial.print(F(": BETINA (HEN) | Conf: "));
    } else if (cls == 1) {
        Serial.print(F(": JANTAN (COCK) | Conf: "));
    } else {
        Serial.print(F(": UNDEFINED | Conf: "));
    }
    Serial.print(cInt);
    Serial.print('.');
    if (cDec < 10U) Serial.print('0');
    Serial.print(cDec);
    Serial.print(F("% | Servo: "));
    Serial.print(g_current_angle);
    Serial.println(F(" deg"));
}

// ============================================================================
// 9. SETUP & MAIN LOOP ARDUINO UNO R3
// ============================================================================
void setup()
{
    Serial.begin(115200);
    delay(100);
    Serial.println(F("=========================================================="));
    Serial.println(F("  PP2 EGG SEXING AI — ARDUINO UNO R3 MASTER CONTROLLER    "));
    Serial.println(F("=========================================================="));

    // 1. Inisialisasi Bus I2C sebagai Slave 0x55 (cukup dipanggil SEKALI di setup!)
    Wire.begin((uint8_t)GROVE_I2C_SLAVE_ADDR);
    Wire.setClock(I2C_BUS_FREQ);
    Wire.onReceive(onGroveReceive);
    digitalWrite(SDA, LOW);
    digitalWrite(SCL, LOW);

    // Aktifkan Hardware I2C Timeout (25 ms) dengan auto-reset agar Wire tidak pernah hang
#if defined(WIRE_HAS_TIMEOUT)
    Wire.setWireTimeout(25000UL, true);
#endif

    // 2. Inisialisasi Layar OLED SSD1306 (0x3C) tanpa mereset Wire (periphBegin = false)
    if (display.begin(SSD1306_SWITCHCAPVCC, OLED_I2C_ADDR, false, false)) {
        g_oled_ready = true;
        Serial.println(F("[INIT] Layar OLED SSD1306 (0x3C) siap di Pin A4 (SDA) & A5 (SCL)"));
    } else {
        g_oled_ready = false;
        Serial.println(F("[WARN] Layar OLED SSD1306 tidak ditemukan di 0x3C!"));
    }
    ensureTwiSlaveReady();

    // 3. Inisialisasi Motor Servo 3-Pin pada Pin 9 (Timer1) & lakukan Self-Test
    eggServo.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
    g_current_state = STATE_BOOT_INIT;
    renderOledState();

    Serial.println(F("[INIT] Self-Test Motor Servo di Pin 9 (90 -> 0 -> 180 -> 90 deg)..."));
    setServoAngle(90);
    delay(400);
    setServoAngle(0);
    delay(550);
    setServoAngle(180);
    delay(550);
    setServoAngle(SERVO_ANGLE_STANDBY);
    delay(450);

    // 4. Masuk ke State Awal (STATE_WAITING_DATA: Servo 90 deg, OLED "WAITING AI...")
    transitionToState(STATE_WAITING_DATA);

    Serial.println(F("[READY] State Awal (WAITING_DATA): Servo di 90 deg (Center Point)."));
    Serial.println(F("        Menunggu paket 5-Byte dari Grove Vision AI V2 di Slave 0x55 (A4/A5)..."));
    Serial.println(F("        (Tips Uji Cepat Serial: ketik '0'=Betina, '1'=Jantan, '2'=Undef)"));
}

void loop()
{
    uint32_t now = millis();

    // 0. Fitur Uji Cepat via Serial Monitor (ketik '0', '1', atau '2' di Serial Monitor)
    if (Serial.available() > 0) {
        char ch = (char)Serial.read();
        if (ch == '0') {
            g_rx_class_id = 0; g_rx_conf_int = 92; g_rx_conf_dec = 50;
            g_packet_count++;  g_new_packet_flag = true;
        } else if (ch == '1') {
            g_rx_class_id = 1; g_rx_conf_int = 95; g_rx_conf_dec = 25;
            g_packet_count++;  g_new_packet_flag = true;
        } else if (ch == '2') {
            g_rx_class_id = 2; g_rx_conf_int = 65; g_rx_conf_dec = 0;
            g_packet_count++;  g_new_packet_flag = true;
        }
    }

    // 1. Cek jika ada error paket I2C
    if (g_packet_err_flag) {
        noInterrupts();
        g_packet_err_flag = false;
        interrupts();

        Serial.println(F("[WARN] Paket I2C dari Grove tidak valid (Checksum/Header Error)!"));
        transitionToState(STATE_I2C_ERROR);
    }

    // 2. Cek jika ada paket 5-Byte baru yang valid dari Grove Vision AI V2
    if (g_new_packet_flag) {
        noInterrupts();
        uint8_t  cls      = g_rx_class_id;
        uint8_t  confInt  = g_rx_conf_int;
        uint8_t  confDec  = g_rx_conf_dec;
        uint16_t pktNum   = g_packet_count;
        g_new_packet_flag = false;
        interrupts();

        g_disp_conf_int = confInt;
        g_disp_conf_dec = confDec;

        // Evaluasi State Machine berdasarkan class_id & confidence dari Grove Vision AI V2
        if (cls == 0) {
            transitionToState(STATE_SORT_BETINA);
        } else if (cls == 1) {
            transitionToState(STATE_SORT_JANTAN);
        } else {
            if (confInt == 0 && confDec == 0) {
                transitionToState(STATE_UNDEFINED_NO_EGG);
            } else {
                transitionToState(STATE_UNDEFINED_LOW_CONF);
            }
        }
        logClassificationSerial(pktNum, cls, confInt, confDec);
        return;
    }

    // 3. Cek Cooldown untuk State Sortir (Betina 0 deg / Jantan 180 deg)
    //    Setelah Cooldown selesai, otomatis kembalikan Servo ke Center Point (90 deg)
    //    dan kembalikan OLED ke tampilan State Awal ("STATUS: WAITING AI...")
    if (g_current_state == STATE_SORT_BETINA || g_current_state == STATE_SORT_JANTAN) {
        if ((uint32_t)(now - g_state_enter_ms) >= SORT_COOLDOWN_MS) {
            Serial.println(F("[AUTO-RESET] Cooldown selesai -> Servo kembali ke 90 deg & OLED kembali ke WAITING AI."));
            transitionToState(STATE_WAITING_DATA);
        }
    }
    // 4. Cek Cooldown untuk State Undefined (Frame Kosong / Low Confidence)
    else if (g_current_state == STATE_UNDEFINED_NO_EGG || g_current_state == STATE_UNDEFINED_LOW_CONF) {
        if ((uint32_t)(now - g_state_enter_ms) >= UNDEF_COOLDOWN_MS) {
            Serial.println(F("[AUTO-RESET] Undefined selesai -> Kembali ke WAITING AI (Servo 90 deg)."));
            transitionToState(STATE_WAITING_DATA);
        }
    }
    // 5. Auto-recovery dari STATE_I2C_ERROR kembali ke STATE_WAITING_DATA
    else if (g_current_state == STATE_I2C_ERROR) {
        if ((uint32_t)(now - g_state_enter_ms) >= ERROR_COOLDOWN_MS) {
            transitionToState(STATE_WAITING_DATA);
        }
    }

    // Pastikan register hardware TWI Slave 0x55 selalu terjaga di setiap siklus idle
    ensureTwiSlaveReady();
    delay(10);
}
