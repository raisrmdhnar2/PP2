/*
 * main_fix.ino — Firmware Master Controller Arduino Uno R3 (PP2 Egg Sexing AI)
 *
 * Target Board: "Arduino Uno" (ATmega328P @ 16 MHz)
 * Lokasi File : PP2/firmware/main_fix/main_fix.ino
 *
 * ARSITEKTUR SESUAI SKEMATIK HARDWARE:
 *   - Arduino Uno terhubung ke Logic Level Converter (U3) sisi HV (5V):
 *       Pin 18 (A4) -> SDA1 (HV2)
 *       Pin 19 (A5) -> SCL1 (HV1)
 *   - Sisi LV (3.3V) terhubung ke:
 *       Grove Vision AI V2 (PA3 -> SDA2, PA2 -> SCL2)
 *
 * PERBAIKAN KRUSIAL I2C:
 *   1. Menjadikan Arduino Uno sebagai PURE I2C SLAVE (0x55) tanpa pernah memanggil
 *      Wire.beginTransmission() yang menyebabkan driver TWI AVR terkunci di Master Mode.
 *   2. Menghapus perintah digitalWrite(SDA, LOW) yang sebelumnya mengganggu
 *      resistor pull-up 10k pada Logic Level Converter U3.
 *   3. Menjamin respon ACK 100% instan ke Grove Vision AI V2 di setiap inferensi.
 *   4. Kendali Motor Servo MG di Pin 9 (PWM Timer1) dengan Cooldown 4 detik.
 */

#include <Arduino.h>
#include <Wire.h>
#include <Servo.h>
#include <Adafruit_NeoPixel.h>

// ============================================================================
// 1. KONFIGURASI PERANGKAT KERAS (ARDUINO UNO R3)
// ============================================================================

// Alamat I2C Slave Arduino Uno (Target pengiriman dari Grove Vision AI V2)
#define GROVE_I2C_SLAVE_ADDR  0x55
#define I2C_BUS_FREQ          100000UL

// Pin LED Indikator I2C (LED "L" bawaan papan Arduino Uno pada Pin 13)
#define LED_I2C_STATUS        LED_BUILTIN // Pin 13

// Konfigurasi Lampu WS2812 / NeoPixel Ring (Pin D6)
// Wiring Fisik:
//   - GND -> GND Arduino / GND Power
//   - 5V  -> 5V Power Supply / 5V Arduino
//   - DI / D1 (Data In) -> Pin D6 Arduino
#define WS2812_PIN            6       // Pin data digital D6 (terhubung ke DI/D1 WS2812)
#define WS2812_NUM_LEDS       24      // Jumlah mata LED ring (mendukung 8, 12, 16, hingga 24 LED)
#define WS2812_BRIGHTNESS     255     // Kecerahan maksimal (0-255)

// Konfigurasi Motor Servo MG (Metal Gear) pada Pin 9
#define SERVO_PIN             9
#define SERVO_MIN_US          500     // Pulsa 0 derajat servo MG (500 us)
#define SERVO_MAX_US          2500    // Pulsa 180 derajat servo MG (2500 us)

// Sudut Gerak Servo Pemilah
#define SERVO_ANGLE_BETINA    50      // 45 derajat  -> Jalur Sortir Kiri (Betina)
#define SERVO_ANGLE_STANDBY   90      // 90 derajat  -> Posisi Netral / Standby
#define SERVO_ANGLE_JANTAN    125     // 135 derajat -> Jalur Sortir Kanan (Jantan)

// Durasi Tahan (Cooldown) Sebelum Servo Kembali ke Center (90°)
#define SORT_COOLDOWN_MS      4000UL  // 4 detik tahan sortir lalu auto-reset ke 90°
#define UNDEF_COOLDOWN_MS     2500UL  // 2,5 detik tampil Undefined

// Protokol Paket dari Grove Vision AI V2
#define PP2_PKT_HEADER        0xAA
#define PP2_PKT_SIZE          5

// ============================================================================
// 2. DEFINISI STATE MACHINE CONTROLLER
// ============================================================================
typedef enum {
    STATE_BOOT_INIT = 0,
    STATE_WAITING_DATA,         // Standby: Servo 90° (Tengah), Menunggu hasil deteksi AI
    STATE_SORT_BETINA,          // Deteksi Betina: Servo 0° (Kiri), Cooldown 4 detik
    STATE_SORT_JANTAN,          // Deteksi Jantan: Servo 180° (Kanan), Cooldown 4 detik
    STATE_UNDEFINED             // Undefined: Servo 90°
} uno_state_t;

// ============================================================================
// 3. OBJEK & VARIABEL GLOBAL
// ============================================================================
Servo eggServo;
Adafruit_NeoPixel ledRing(WS2812_NUM_LEDS, WS2812_PIN, NEO_GRB + NEO_KHZ800);

volatile bool     g_i2c_activity_flag = false;
volatile bool     g_new_packet_flag   = false;
volatile bool     g_packet_err_flag   = false;
volatile uint8_t  g_rx_class_id       = 2;
volatile uint8_t  g_rx_conf_int       = 0;
volatile uint8_t  g_rx_conf_dec       = 0;
volatile uint16_t g_packet_count      = 0;

uno_state_t g_current_state           = STATE_BOOT_INIT;
uint8_t     g_current_angle           = SERVO_ANGLE_STANDBY;
uint32_t    g_state_enter_ms          = 0;

// Callback deklarasi
void onGroveReceive(int numBytes);

// ============================================================================
// 4. FUNGSI KONTROL MOTOR SERVO MG (PIN 9 TIMER1)
// ============================================================================
void setServoAngle(uint8_t targetAngle)
{
    if (targetAngle > 180) targetAngle = 180;
    g_current_angle = targetAngle;
    eggServo.write(g_current_angle);

    Serial.print(F("[SERVO MG] Pin 9 -> Sudut: "));
    Serial.print(g_current_angle);
    Serial.println(F(" deg"));
}

// Helper Lampu WS2812 NeoPixel Ring (Pin D6)
void setWS2812Color(uint8_t r, uint8_t g, uint8_t b)
{
    for (uint16_t i = 0; i < WS2812_NUM_LEDS; i++) {
        ledRing.setPixelColor(i, ledRing.Color(r, g, b));
    }
    ledRing.show();
}

// ============================================================================
// 5. INTERRUPT CALLBACK I2C SLAVE (MENERIMA 5 BYTE DARI GROVE VISION AI V2)
// ============================================================================
void onGroveReceive(int numBytes)
{
    // Indikator Visual Fisik: Toggle LED "L" (Pin 13) seketika saat ada sinyal I2C masuk!
    digitalWrite(LED_I2C_STATUS, !digitalRead(LED_I2C_STATUS));
    g_i2c_activity_flag = true;

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

    // Paket valid dari Grove Vision AI V2 berhasil diterima!
    g_rx_class_id     = pkt[1];
    g_rx_conf_int     = pkt[2];
    g_rx_conf_dec     = pkt[3];
    g_packet_count    = g_packet_count + 1;
    g_new_packet_flag = true;
}

// ============================================================================
// 6. TRANSISI STATE MACHINE ARDUINO UNO
// ============================================================================
void transitionToState(uno_state_t nextState)
{
    g_current_state = nextState;

    if (nextState == STATE_SORT_BETINA) {
        setServoAngle(SERVO_ANGLE_BETINA);
        g_state_enter_ms = millis();
    }
    else if (nextState == STATE_SORT_JANTAN) {
        setServoAngle(SERVO_ANGLE_JANTAN);
        g_state_enter_ms = millis();
    }
    else if (nextState == STATE_WAITING_DATA) {
        setServoAngle(SERVO_ANGLE_STANDBY);
        g_state_enter_ms = millis();
    }
    else {
        setServoAngle(SERVO_ANGLE_STANDBY);
        g_state_enter_ms = millis();
    }
}

// Helper log Serial Monitor
static void logClassificationSerial(uint16_t pktNum, uint8_t cls, uint8_t cInt, uint8_t cDec)
{
    Serial.print(F("[#"));
    Serial.print(pktNum);
    Serial.print(F("] GROVE AI I2C [0x55] -> CLASS "));
    Serial.print(cls);
    if (cls == 0) {
        Serial.print(F(": BETINA (HEN) | Akurasi: "));
    } else if (cls == 1) {
        Serial.print(F(": JANTAN (COCK) | Akurasi: "));
    } else {
        Serial.print(F(": UNDEFINED | Akurasi: "));
    }
    Serial.print(cInt);
    Serial.print('.');
    if (cDec < 10U) Serial.print('0');
    Serial.print(cDec);
    Serial.print(F("% | Servo MG: "));
    Serial.print(g_current_angle);
    Serial.println(F(" deg"));
}

// ============================================================================
// 7. SETUP & MAIN LOOP ARDUINO UNO R3
// ============================================================================
void setup()
{
    Serial.begin(115200);
    delay(100);
    Serial.println(F("\n========================================================"));
    Serial.println(F("  PP2 EGG SEXING — ARDUINO UNO R3 (I2C SLAVE 0x55)     "));
    Serial.println(F("  Level Converter U3: A4 (SDA1) & A5 (SCL1)            "));
    Serial.println(F("========================================================"));

    // 0. Inisialisasi Lampu WS2812 NeoPixel Ring (Pin D6) - AKTIF PUTIH NYALA TERUS
    ledRing.begin();
    ledRing.clear();
    ledRing.show();
    delay(50); // Latch reset WS2812
    ledRing.setBrightness(WS2812_BRIGHTNESS);
    setWS2812Color(255, 255, 255); // Putih Terang untuk Penerangan Optimal Kamera AI
    Serial.println(F("[WS2812] Lampu Penerangan Kamera (Pin D6) -> AKTIF PUTIH (NYALA TERUS)"));

    // 1. Inisialisasi LED Indikator I2C (LED "L" Pin 13)
    pinMode(LED_I2C_STATUS, OUTPUT);
    digitalWrite(LED_I2C_STATUS, HIGH); // LED ON = Slave 0x55 Init Selesai & Siap!

    // 2. Inisialisasi PURE I2C SLAVE pada Alamat 0x55
    Wire.begin((uint8_t)GROVE_I2C_SLAVE_ADDR);
    Wire.setClock(I2C_BUS_FREQ);
    Wire.onReceive(onGroveReceive);

    // 3. Inisialisasi Motor Servo MG pada Pin 9
    eggServo.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
    g_current_state = STATE_BOOT_INIT;

    Serial.println(F("[INIT] Self-Test Motor Servo MG (90 -> 45 -> 135 -> 90 deg)..."));
    setServoAngle(SERVO_ANGLE_STANDBY);
    delay(400);
    setServoAngle(SERVO_ANGLE_BETINA);
    delay(550);
    setServoAngle(SERVO_ANGLE_JANTAN);
    delay(550);
    setServoAngle(SERVO_ANGLE_STANDBY);
    delay(450);

    // 4. Masuk ke State Standby
    transitionToState(STATE_WAITING_DATA);

    Serial.println(F("[READY] I2C Slave Siap di Alamat: 0x55 (LED L Pin 13 = ON)"));
    Serial.println(F("        Menunggu paket I2C dari Grove Vision AI V2..."));
    Serial.println(F("        (Uji Cepat Serial: ketik '0'=Betina, '1'=Jantan, '9'=Standby)"));
}

void loop()
{
    uint32_t now = millis();

    // 0. Uji Cepat via Serial Monitor
    if (Serial.available() > 0) {
        char ch = (char)Serial.read();
        if (ch == '0') {
            g_rx_class_id = 0; g_rx_conf_int = 95; g_rx_conf_dec = 0;
            g_packet_count++;  g_new_packet_flag = true;
        } else if (ch == '1') {
            g_rx_class_id = 1; g_rx_conf_int = 97; g_rx_conf_dec = 50;
            g_packet_count++;  g_new_packet_flag = true;
        } else if (ch == '9') {
            transitionToState(STATE_WAITING_DATA);
        } else if (ch == 'w' || ch == 'W') {
            setWS2812Color(255, 255, 255);
            Serial.println(F("[WS2812] Lampu Putih D6 di-refresh!"));
        }
    }

    // 0b. Deteksi Sinyal Fisik Masuk ke I2C Slave
    if (g_i2c_activity_flag) {
        noInterrupts();
        g_i2c_activity_flag = false;
        interrupts();
        Serial.println(F("[I2C PULSE] Sinyal detak fisik I2C terdeteksi masuk dari Grove!"));
    }

    // 1. Deteksi Error Checksum
    if (g_packet_err_flag) {
        noInterrupts();
        g_packet_err_flag = false;
        interrupts();
        Serial.println(F("[WARN] Paket I2C diterima tapi Checksum Salah!"));
    }

    // 2. Paket Valid Baru Diterima dari Grove Vision AI V2
    if (g_new_packet_flag) {
        noInterrupts();
        uint8_t  cls      = g_rx_class_id;
        uint8_t  confInt  = g_rx_conf_int;
        uint8_t  confDec  = g_rx_conf_dec;
        uint16_t pktNum   = g_packet_count;
        g_new_packet_flag = false;
        interrupts();

        // Gerakkan Servo Sesuai Keputusan AI
        if (cls == 0) {
            transitionToState(STATE_SORT_BETINA);
        } else if (cls == 1) {
            transitionToState(STATE_SORT_JANTAN);
        } else {
            transitionToState(STATE_UNDEFINED);
        }
        logClassificationSerial(pktNum, cls, confInt, confDec);
        return;
    }

    // 3. Auto-Reset Cooldown 4 Detik
    if (g_current_state == STATE_SORT_BETINA || g_current_state == STATE_SORT_JANTAN) {
        if ((uint32_t)(now - g_state_enter_ms) >= SORT_COOLDOWN_MS) {
            Serial.println(F("[RESET] Cooldown 4s selesai -> Servo kembali ke 90 deg (Standby)."));
            transitionToState(STATE_WAITING_DATA);
        }
    }
    else if (g_current_state == STATE_UNDEFINED) {
        if ((uint32_t)(now - g_state_enter_ms) >= UNDEF_COOLDOWN_MS) {
            transitionToState(STATE_WAITING_DATA);
        }
    }

    delay(10);
}
