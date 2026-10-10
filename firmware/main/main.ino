/*
 * main.ino — Firmware Master Controller ESP32-S3 Dev Module (PP2 Egg Sexing AI)
 *
 * Target Board di Arduino IDE: "ESP32S3 Dev Module"
 *
 * Peran ESP32-S3:
 *   1. Menerima paket I2C 5-Byte murni AI dari Seeed Grove Vision AI Module V2:
 *        [Byte 0: 0xAA, Byte 1: class_id, Byte 2: conf_int, Byte 3: conf_dec, Byte 4: checksum]
 *      melalui bus Wire1 (I2C Slave 0x55 pada GPIO 4 SDA & GPIO 5 SCL).
 *   2. Mengambil keputusan sudut aktuator & menggerakkan Motor Servo 3-Pin (GPIO 6):
 *        - Default / Standby (State Pertama) : Servo 90 derajat
 *        - class_id == 0 (Betina / Hen)      : Servo 0 derajat   (Kiri)  -> Tahan/Cooldown 5 detik -> Reset ke 90 derajat
 *        - class_id == 1 (Jantan / Cock)     : Servo 180 derajat (Kanan) -> Tahan/Cooldown 5 detik -> Reset ke 90 derajat
 *        - class_id == 2 (Undefined)         : Servo 90 derajat  (Hold)  -> Tahan/Cooldown 3 detik -> Reset ke State Pertama
 *   3. Menjalankan State Machine tampilan Layar OLED SSD1306 128x64 (0x3C)
 *      melalui bus Wire (I2C Master default ESP32-S3 pada GPIO 8 SDA & GPIO 9 SCL).
 *
 * Catatan Teknis Sinyal PWM Servo:
 *   Menggunakan metode Direct Microsecond Pulse 50 Hz (digitalWrite + delayMicroseconds)
 *   yang identik 100% dengan test_servo.ino yang sudah terbukti berhasil menggerakkan servo
 *   di board ESP32-S3 Anda (bebas dari bug clock divider LEDC 14-bit pada ESP32-S3).
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ============================================================================
// 1. KONFIGURASI PIN & ALAMAT PERANGKAT KERAS (KHUSUS ESP32-S3 DEV MODULE)
// ============================================================================
// Pemetaan Pin Resmi ESP32-S3-DevKitC-1:
//   * Wire1 (Slave dari Grove) : SDA = GPIO 4, SCL = GPIO 5
//   * Wire  (Master ke OLED)   : SDA = GPIO 8, SCL = GPIO 9 (Default I2C ESP32-S3)
//   * Motor Servo 3-Pin        : Signal PWM = GPIO 6 (juga di-mirror ke GPIO 7, 10, 14, 15, 16, 17)
//                                VCC = Baterai Eksternal (+), GND = Baterai (-) + GND ESP32-S3

// Bus I2C_1 (Slave penerima paket 5-Byte dari Grove Vision AI V2)
#define GROVE_I2C_SLAVE_ADDR  0x55
#define GROVE_SDA_PIN         4
#define GROVE_SCL_PIN         5
#define GROVE_I2C_FREQ        100000

// Bus I2C_0 (Master penggerak layar OLED SSD1306 128x64 — Default I2C ESP32-S3)
#define OLED_I2C_ADDR         0x3C
#define OLED_SDA_PIN          8
#define OLED_SCL_PIN          9
#define SCREEN_WIDTH          128
#define SCREEN_HEIGHT         64
#define OLED_RESET            -1

// Konfigurasi Pin & Pulsa Motor Servo 3-Pin (Identik 100% dengan test_servo.ino)
#define SERVO_PWM_PIN         6       // Pin utama Servo = GPIO 6
const int SERVO_MIRROR_PINS[] = {6, 7, 10, 14, 15, 16, 17};
const int NUM_SERVO_PINS      = sizeof(SERVO_MIRROR_PINS) / sizeof(SERVO_MIRROR_PINS[0]);

#define SERVO_MIN_US          600     // Lebar pulsa 0 derajat (us) — batas aman agar gir tidak mentok
#define SERVO_MAX_US          2300    // Lebar pulsa 180 derajat (us) — batas aman agar gir tidak mentok
#define SERVO_PERIOD_US       20000   // Periode 50 Hz = 20.000 us (20 ms)

// Keputusan Sudut Servo yang Dikomandoi oleh ESP32-S3
#define SERVO_ANGLE_BETINA    0       // 0 derajat   -> Jalur Sortir Kiri (Betina)
#define SERVO_ANGLE_STANDBY   90      // 90 derajat  -> Posisi Default / Tengah (Standby)
#define SERVO_ANGLE_JANTAN    180     // 180 derajat -> Jalur Sortir Kanan (Jantan)

// Durasi Cooldown Sebelum Reset Kembali ke State Pertama (STATE_WAITING_DATA / 90 derajat)
#define SORT_COOLDOWN_MS      5000UL  // 5 detik (5000 ms) cooldown setelah gerak Jantan/Betina
#define UNDEF_COOLDOWN_MS     3000UL  // 3 detik (3000 ms) tampil Undefined sebelum kembali ke Waiting
#define ERROR_COOLDOWN_MS     2000UL  // 2 detik (2000 ms) auto-recovery dari error paket I2C

// Protokol Paket 5-Byte dari Grove Vision AI V2
#define PP2_PKT_HEADER        0xAA
#define PP2_PKT_SIZE          5

// ============================================================================
// 2. DEFINISI STATE MACHINE ESP32-S3 (OLED & SERVO)
// ============================================================================
typedef enum {
    STATE_BOOT_INIT = 0,
    STATE_WAITING_DATA,         // State Pertama (Default): Servo 90 derajat, siap terima input
    STATE_SORT_BETINA,          // Aktuasi Betina: Servo 0 derajat selama Cooldown 5 detik
    STATE_SORT_JANTAN,          // Aktuasi Jantan: Servo 180 derajat selama Cooldown 5 detik
    STATE_UNDEFINED_NO_EGG,     // Undefined (Frame Kosong): Servo 90 derajat
    STATE_UNDEFINED_LOW_CONF,   // Undefined (Ragu <80%): Servo 90 derajat
    STATE_I2C_ERROR             // Error paket I2C: Servo 90 derajat
} esp_state_t;

// ============================================================================
// 3. OBJEK & VARIABEL GLOBAL
// ============================================================================
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

volatile bool     g_new_packet_flag   = false;
volatile bool     g_packet_err_flag   = false;
volatile uint8_t  g_rx_class_id       = 2;
volatile uint8_t  g_rx_conf_int       = 0;
volatile uint8_t  g_rx_conf_dec       = 0;
volatile uint32_t g_packet_count      = 0;

esp_state_t g_current_state           = STATE_BOOT_INIT;
bool        g_oled_ready              = false;
int         g_current_angle           = SERVO_ANGLE_STANDBY;
int         g_current_pulse_us        = 1450;
float       g_current_conf            = 0.0f;
uint32_t    g_state_enter_ms          = 0;
uint32_t    g_last_countdown_sec      = 0;

// ============================================================================
// 4. FUNGSI PEMBANGKIT PULSA SERVO 50 Hz (IDENTIK DENGAN test_servo.ino)
// ============================================================================
void initServoPins()
{
    for (int i = 0; i < NUM_SERVO_PINS; ++i) {
        pinMode(SERVO_MIRROR_PINS[i], OUTPUT);
        digitalWrite(SERVO_MIRROR_PINS[i], LOW);
    }
}

// Mengirimkan 1 siklus pulsa 50 Hz (20 ms) ke GPIO 6 (dan pin mirror)
void sendOneServoPulse()
{
    int pulseUs = g_current_pulse_us;
    int lowUs   = SERVO_PERIOD_US - pulseUs;

    for (int i = 0; i < NUM_SERVO_PINS; ++i) {
        digitalWrite(SERVO_MIRROR_PINS[i], HIGH);
    }
    delayMicroseconds(pulseUs);

    for (int i = 0; i < NUM_SERVO_PINS; ++i) {
        digitalWrite(SERVO_MIRROR_PINS[i], LOW);
    }
    delayMicroseconds(lowUs);
}

// Mengatur sudut target dan langsung mengirimkan rentetan pulsa awal (burst)
// agar tuas servo langsung bergerak tegas ke sudut tujuan
void setServoAngle(int targetAngle, uint32_t burstMs = 400)
{
    if (targetAngle < 0) targetAngle = 0;
    if (targetAngle > 180) targetAngle = 180;

    g_current_angle    = targetAngle;
    g_current_pulse_us = SERVO_MIN_US + ((targetAngle * (SERVO_MAX_US - SERVO_MIN_US)) / 180);

    Serial.printf("[SERVO] GPIO %d -> Sudut: %3d deg (Pulsa: %4d us)\n",
                  SERVO_PWM_PIN, g_current_angle, g_current_pulse_us);

    uint32_t numPulses = burstMs / 20U;
    if (numPulses == 0) numPulses = 1;
    for (uint32_t p = 0; p < numPulses; ++p) {
        sendOneServoPulse();
    }
}

// ============================================================================
// 5. INTERRUPT CALLBACK I2C SLAVE (MENERIMA 5 BYTE DARI GROVE VISION AI V2)
// ============================================================================
void onGroveReceive(int numBytes)
{
    if (numBytes < PP2_PKT_SIZE) {
        while (Wire1.available()) {
            Wire1.read();
        }
        return;
    }

    uint8_t pkt[PP2_PKT_SIZE];
    for (int i = 0; i < PP2_PKT_SIZE; ++i) {
        pkt[i] = (uint8_t)Wire1.read();
    }
    while (Wire1.available()) {
        Wire1.read();
    }

    // Validasi Header (0xAA) & XOR Checksum
    uint8_t calc_crc = (uint8_t)(pkt[0] ^ pkt[1] ^ pkt[2] ^ pkt[3]);
    if (pkt[0] != PP2_PKT_HEADER || pkt[4] != calc_crc || pkt[1] > 2) {
        g_packet_err_flag = true;
        return;
    }

    // Abaikan paket baru jika sedang berada dalam masa Cooldown Sortir (Betina / Jantan)
    // agar gerakan mekanik servo & hitung mundur 5 detik tidak terganggu di tengah jalan
    if (g_current_state == STATE_SORT_BETINA || g_current_state == STATE_SORT_JANTAN) {
        return;
    }

    g_rx_class_id     = pkt[1];
    g_rx_conf_int     = pkt[2];
    g_rx_conf_dec     = pkt[3];
    g_packet_count    = g_packet_count + 1;
    g_new_packet_flag = true;
}

// ============================================================================
// 6. FUNGSI RENDERING STATE MACHINE LAYAR OLED SSD1306 (128x64)
// ============================================================================

static void drawConfidenceBar(float confPct)
{
    const int barX = 4;
    const int barY = 41;
    const int barW = 120;
    const int barH = 9;

    display.drawRect(barX, barY, barW, barH, SSD1306_WHITE);

    float clamped = confPct;
    if (clamped < 0.0f) clamped = 0.0f;
    if (clamped > 100.0f) clamped = 100.0f;

    int fillW = (int)((clamped / 100.0f) * (float)(barW - 4) + 0.5f);
    if (fillW > 0) {
        display.fillRect(barX + 2, barY + 2, fillW, barH - 4, SSD1306_WHITE);
    }

    // Garis vertikal penanda Threshold 80%
    int thX = barX + 2 + (int)(0.80f * (float)(barW - 4));
    display.drawLine(thX, barY - 2, thX, barY + barH + 1, SSD1306_INVERSE);
}

static void drawHeaderBanner(uint32_t pktCount)
{
    display.fillRect(0, 0, SCREEN_WIDTH, 11, SSD1306_WHITE);
    display.setTextSize(1);
    display.setTextColor(SSD1306_BLACK);
    display.setCursor(3, 2);
    char hdrBuf[24];
    snprintf(hdrBuf, sizeof(hdrBuf), "PP2 EGG SEXING [#%03lu]", (unsigned long)(pktCount % 1000));
    display.print(hdrBuf);
    display.setTextColor(SSD1306_WHITE);
}

static uint32_t getRemainingCooldownSec(uint32_t cooldownDurationMs)
{
    uint32_t elapsed = millis() - g_state_enter_ms;
    if (elapsed >= cooldownDurationMs) return 0;
    return (cooldownDurationMs - elapsed + 999UL) / 1000UL;
}

void renderOledState()
{
    if (!g_oled_ready) return;

    display.clearDisplay();

    switch (g_current_state) {
        case STATE_BOOT_INIT: {
            display.fillRect(0, 0, SCREEN_WIDTH, 12, SSD1306_WHITE);
            display.setTextSize(1);
            display.setTextColor(SSD1306_BLACK);
            display.setCursor(5, 2);
            display.print("PP2 ESP32-S3 CONTROL");

            display.setTextColor(SSD1306_WHITE);
            display.setCursor(4, 16);
            display.print("SELF-TEST SERVO...");
            display.setCursor(4, 28);
            display.print("I2C Slv: 0x55 (G4/G5)");
            display.setCursor(4, 40);
            display.print("OLED   : 0x3C (G8/G9)");
            display.setCursor(4, 52);
            display.print("Servo  : 90 DEG (G6)");
            break;
        }

        case STATE_WAITING_DATA: {
            drawHeaderBanner(g_packet_count);
            display.setCursor(4, 15);
            display.print("STATUS: WAITING AI...");
            display.setCursor(4, 28);
            display.print("CONF  : --.-- %");
            drawConfidenceBar(0.0f);
            display.setCursor(4, 54);
            display.print("SERVO : 90 DEG (READY)");
            break;
        }

        case STATE_SORT_BETINA: {
            uint32_t remSec = getRemainingCooldownSec(SORT_COOLDOWN_MS);
            drawHeaderBanner(g_packet_count);
            display.setCursor(4, 15);
            display.print("HASIL : BETINA (HEN)");
            display.setCursor(4, 28);
            char confStr[24];
            snprintf(confStr, sizeof(confStr), "CONF  : %.2f %%", g_current_conf);
            display.print(confStr);
            drawConfidenceBar(g_current_conf);
            display.setCursor(4, 54);
            char srvStr[26];
            snprintf(srvStr, sizeof(srvStr), "SERVO: 0 DEG [RST %lus]", (unsigned long)remSec);
            display.print(srvStr);
            break;
        }

        case STATE_SORT_JANTAN: {
            uint32_t remSec = getRemainingCooldownSec(SORT_COOLDOWN_MS);
            drawHeaderBanner(g_packet_count);
            display.setCursor(4, 15);
            display.print("HASIL : JANTAN (COCK)");
            display.setCursor(4, 28);
            char confStr[24];
            snprintf(confStr, sizeof(confStr), "CONF  : %.2f %%", g_current_conf);
            display.print(confStr);
            drawConfidenceBar(g_current_conf);
            display.setCursor(4, 54);
            char srvStr[26];
            snprintf(srvStr, sizeof(srvStr), "SERVO:180 DEG[RST %lus]", (unsigned long)remSec);
            display.print(srvStr);
            break;
        }

        case STATE_UNDEFINED_NO_EGG: {
            uint32_t remSec = getRemainingCooldownSec(UNDEF_COOLDOWN_MS);
            drawHeaderBanner(g_packet_count);
            display.setCursor(4, 15);
            display.print("HASIL : UNDEFINED");
            display.setCursor(4, 28);
            display.print("CONF  : 0.00% (NO EGG)");
            drawConfidenceBar(0.0f);
            display.setCursor(4, 54);
            char srvStr[26];
            snprintf(srvStr, sizeof(srvStr), "SERVO: 90 DEG[RST %lus]", (unsigned long)remSec);
            display.print(srvStr);
            break;
        }

        case STATE_UNDEFINED_LOW_CONF: {
            uint32_t remSec = getRemainingCooldownSec(UNDEF_COOLDOWN_MS);
            drawHeaderBanner(g_packet_count);
            display.setCursor(4, 15);
            display.print("HASIL : UNDEF (<80%)");
            display.setCursor(4, 28);
            char confStr[24];
            snprintf(confStr, sizeof(confStr), "CONF  : %.2f%% (RAGU)", g_current_conf);
            display.print(confStr);
            drawConfidenceBar(g_current_conf);
            display.setCursor(4, 54);
            char srvStr[26];
            snprintf(srvStr, sizeof(srvStr), "SERVO: 90 DEG[RST %lus]", (unsigned long)remSec);
            display.print(srvStr);
            break;
        }

        case STATE_I2C_ERROR: {
            drawHeaderBanner(g_packet_count);
            display.setCursor(4, 18);
            display.print("WARN  : I2C PKT ERROR");
            display.setCursor(4, 32);
            display.print("Check GND/SDA/SCL!");
            display.setCursor(4, 54);
            display.print("SERVO : 90 DEG (SAFE)");
            break;
        }
    }

    display.display();
}

// ============================================================================
// 7. TRANSISI STATE MACHINE & EKSEKUSI INSTRUKSI SERVO + OLED
// ============================================================================
void transitionToState(esp_state_t nextState)
{
    g_current_state  = nextState;
    g_state_enter_ms = millis();

    int targetAngle  = SERVO_ANGLE_STANDBY;
    uint32_t burstMs = 350;

    switch (g_current_state) {
        case STATE_SORT_BETINA:
            targetAngle = SERVO_ANGLE_BETINA;   // 0 derajat (Kiri)
            burstMs     = 500;
            g_last_countdown_sec = getRemainingCooldownSec(SORT_COOLDOWN_MS);
            break;

        case STATE_SORT_JANTAN:
            targetAngle = SERVO_ANGLE_JANTAN;   // 180 derajat (Kanan)
            burstMs     = 500;
            g_last_countdown_sec = getRemainingCooldownSec(SORT_COOLDOWN_MS);
            break;

        case STATE_UNDEFINED_NO_EGG:
        case STATE_UNDEFINED_LOW_CONF:
            targetAngle = SERVO_ANGLE_STANDBY;  // 90 derajat (Standby)
            burstMs     = 250;
            g_last_countdown_sec = getRemainingCooldownSec(UNDEF_COOLDOWN_MS);
            break;

        case STATE_BOOT_INIT:
        case STATE_WAITING_DATA:
        case STATE_I2C_ERROR:
        default:
            targetAngle = SERVO_ANGLE_STANDBY;  // 90 derajat (Default Standby)
            burstMs     = 350;
            g_last_countdown_sec = 0;
            break;
    }

    // Update tampilan OLED terlebih dahulu, lalu gerakkan servo dengan pulsa 50 Hz
    renderOledState();
    setServoAngle(targetAngle, burstMs);
    // Reset timer cooldown tepat setelah servo selesai bergerak ke posisi target
    // agar waktu tunggu 5 detik dihitung penuh sejak servo tiba di sudut 0 / 180 derajat
    g_state_enter_ms = millis();
}

// ============================================================================
// 8. SETUP & MAIN LOOP ESP32-S3
// ============================================================================
void setup()
{
    Serial.begin(115200);
    delay(200);
    Serial.println("==========================================================");
    Serial.println("  PP2 EGG SEXING AI — ESP32-S3 MASTER CONTROLLER          ");
    Serial.println("==========================================================");

    // 1. Inisialisasi Pin Output Servo (GPIO 6 + Mirror Pins)
    initServoPins();

    // 2. Inisialisasi Bus I2C_0 (Master) untuk OLED SSD1306 (GPIO 8 SDA, GPIO 9 SCL pada ESP32-S3)
    Wire.begin(OLED_SDA_PIN, OLED_SCL_PIN, 400000);
    if (display.begin(SSD1306_SWITCHCAPVCC, OLED_I2C_ADDR)) {
        g_oled_ready = true;
        Serial.printf("[INIT] Layar OLED SSD1306 (0x%02X) siap di GPIO %d (SDA) & GPIO %d (SCL)\n",
                      OLED_I2C_ADDR, OLED_SDA_PIN, OLED_SCL_PIN);
    } else {
        g_oled_ready = false;
        Serial.println("[WARN] Layar OLED SSD1306 tidak ditemukan di alamat 0x3C!");
    }

    // 3. Tampilkan layar Boot & lakukan Self-Test Gerakan Servo (90 -> 0 -> 180 -> 90 deg)
    g_current_state = STATE_BOOT_INIT;
    renderOledState();
    Serial.printf("[INIT] Self-Test Motor Servo di GPIO %d (90 -> 0 -> 180 -> 90 deg)...\n", SERVO_PWM_PIN);
    setServoAngle(90,  500);
    setServoAngle(0,   600);
    setServoAngle(180, 600);
    setServoAngle(90,  500);

    // 4. Inisialisasi Bus I2C_1 (Slave 0x55) di GPIO 4 (SDA) & GPIO 5 (SCL) untuk Grove Vision AI V2
    pinMode(GROVE_SDA_PIN, INPUT_PULLUP);
    pinMode(GROVE_SCL_PIN, INPUT_PULLUP);
    Wire1.onReceive(onGroveReceive);
    bool slaveOk = Wire1.begin((uint8_t)GROVE_I2C_SLAVE_ADDR, GROVE_SDA_PIN, GROVE_SCL_PIN, GROVE_I2C_FREQ);
    if (slaveOk) {
        Serial.printf("[INIT] I2C Slave (0x%02X) aktif di GPIO %d (SDA) & GPIO %d (SCL)\n",
                      GROVE_I2C_SLAVE_ADDR, GROVE_SDA_PIN, GROVE_SCL_PIN);
    } else {
        Serial.println("[ERROR] Gagal menginisialisasi Wire1 sebagai I2C Slave!");
    }

    // 5. Masuk ke State Pertama (Default: Servo 90 derajat, menunggu input AI)
    transitionToState(STATE_WAITING_DATA);
    Serial.println("[READY] State Pertama (WAITING_DATA): Servo di 90 deg. Menunggu paket 5-Byte dari Grove...");
}

void loop()
{
    uint32_t now = millis();

    // 1. Cek jika ada error paket I2C
    if (g_packet_err_flag) {
        noInterrupts();
        g_packet_err_flag = false;
        interrupts();

        Serial.println("[WARN] Paket I2C dari Grove tidak valid (Checksum/Header Error)!");
        transitionToState(STATE_I2C_ERROR);
    }

    // 2. Cek Cooldown 5 Detik untuk State Sortir (Betina 0 deg / Jantan 180 deg)
    //    Setelah 5 detik selesai, otomatis reset ke State Pertama (STATE_WAITING_DATA, Servo kembali ke 90 deg)
    if (g_current_state == STATE_SORT_BETINA || g_current_state == STATE_SORT_JANTAN) {
        uint32_t elapsed = now - g_state_enter_ms;
        if (elapsed >= SORT_COOLDOWN_MS) {
            Serial.println("[COOLDOWN SELESAI] 5 detik berlalu -> Reset Servo ke 90 deg & kembali ke STATE_WAITING_DATA.");
            transitionToState(STATE_WAITING_DATA);
        } else {
            uint32_t remSec = getRemainingCooldownSec(SORT_COOLDOWN_MS);
            if (remSec != g_last_countdown_sec) {
                g_last_countdown_sec = remSec;
                renderOledState();
            }
        }
    }
    // 3. Cek Cooldown 3 Detik untuk State Undefined (Frame Kosong / Low Confidence)
    else if (g_current_state == STATE_UNDEFINED_NO_EGG || g_current_state == STATE_UNDEFINED_LOW_CONF) {
        uint32_t elapsed = now - g_state_enter_ms;
        if (elapsed >= UNDEF_COOLDOWN_MS) {
            Serial.println("[RESET] Kembali ke STATE_WAITING_DATA (Servo 90 deg).");
            transitionToState(STATE_WAITING_DATA);
        } else {
            uint32_t remSec = getRemainingCooldownSec(UNDEF_COOLDOWN_MS);
            if (remSec != g_last_countdown_sec) {
                g_last_countdown_sec = remSec;
                renderOledState();
            }
        }
    }
    // 4. Auto-recovery dari STATE_I2C_ERROR kembali ke STATE_WAITING_DATA setelah 2 detik
    else if (g_current_state == STATE_I2C_ERROR && (now - g_state_enter_ms >= ERROR_COOLDOWN_MS)) {
        transitionToState(STATE_WAITING_DATA);
    }

    // 5. Cek jika ada paket 5-Byte baru yang valid dari Grove Vision AI V2
    if (g_new_packet_flag) {
        noInterrupts();
        uint8_t cls       = g_rx_class_id;
        uint8_t confInt   = g_rx_conf_int;
        uint8_t confDec   = g_rx_conf_dec;
        uint32_t pktNum   = g_packet_count;
        g_new_packet_flag = false;
        interrupts();

        g_current_conf = (float)confInt + ((float)confDec / 100.0f);

        // Evaluasi State Machine berdasarkan class_id & confidence dari Grove Vision AI V2
        if (cls == 0) {
            transitionToState(STATE_SORT_BETINA);
            Serial.printf("[#%03lu] GROVE AI -> CLASS 0: BETINA (HEN) | Conf: %.2f%% | Servo: %d deg (Kiri) [Cooldown 5s]\n",
                          (unsigned long)pktNum, g_current_conf, g_current_angle);
        } else if (cls == 1) {
            transitionToState(STATE_SORT_JANTAN);
            Serial.printf("[#%03lu] GROVE AI -> CLASS 1: JANTAN (COCK) | Conf: %.2f%% | Servo: %d deg (Kanan) [Cooldown 5s]\n",
                          (unsigned long)pktNum, g_current_conf, g_current_angle);
        } else {
            if (confInt == 0 && confDec == 0) {
                transitionToState(STATE_UNDEFINED_NO_EGG);
                Serial.printf("[#%03lu] GROVE AI -> CLASS 2: UNDEFINED (NO EGG / BLANK) | Conf: 0.00%% | Servo: %d deg (Hold)\n",
                              (unsigned long)pktNum, g_current_angle);
            } else {
                transitionToState(STATE_UNDEFINED_LOW_CONF);
                Serial.printf("[#%03lu] GROVE AI -> CLASS 2: UNDEFINED (LOW CONF <80%%) | Conf: %.2f%% | Servo: %d deg (Hold)\n",
                              (unsigned long)pktNum, g_current_conf, g_current_angle);
            }
        }
    }

    // 6. Kirimkan 1 pulsa 50 Hz (20 ms) secara kontinyu di setiap siklus loop
    //    agar posisi servo selalu terkunci kuat (sama persis dengan test_servo.ino)
    sendOneServoPulse();
}
