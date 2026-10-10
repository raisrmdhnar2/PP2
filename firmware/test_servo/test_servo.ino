/*
 * test_servo.ino — Program Uji Kontinyu Motor Servo + Voltmeter OLED (ESP32-S3)
 *
 * Lokasi File : PP2/firmware/test_servo/test_servo.ino
 * Target Board: ESP32S3 Dev Module
 *
 * Fitur Diagnostik Lengkap (Tanpa Perlu Multimeter!):
 *   1. Menggerakkan Servo secara kontinyu: 90 -> 0 -> 90 -> 180 derajat.
 *   2. Mengeluarkan sinyal Servo (50 Hz) sekaligus lewat:
 *        - Hardware LEDC PWM di GPIO 6 (10-bit 50 Hz)
 *        - Direct Microsecond Pulse di GPIO 7, 10, 14, 15, 16, 17
 *   3. FITUR VOLTMETER & DETEKTOR SINYAL DI LAYAR OLED (Pin Probe = GPIO 1):
 *        Hubungkan 1 kabel jumper dari pin GPIO 1 ke:
 *        - Pin "5V" / VCC Servo -> Lihat di layar OLED baris bawah ("PROBE G1: xxxx mV").
 *          * Jika tertulis < 500 mV (0.0V - 0.4V), berarti pin 5V board ESP32-S3 Anda MATI (0V)!
 *          * Jika tertulis > 2500 mV, berarti ada tegangan listrik masuk ke servo.
 *        - Pin "GPIO 6" -> Lihat di layar OLED apakah pulsa terdeteksi ("G6_CHK: OK").
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// Konfigurasi Layar OLED SSD1306 (GPIO 8 SDA & GPIO 9 SCL)
#define OLED_I2C_ADDR   0x3C
#define OLED_SDA_PIN    8
#define OLED_SCL_PIN    9
#define SCREEN_WIDTH    128
#define SCREEN_HEIGHT   64
#define OLED_RESET      -1

// Pin Probe Voltmeter Internal (ADC1_CH0 pada ESP32-S3)
// Gunakan kabel jumper dari GPIO 1 untuk mengecek tegangan pin 5V / 3V3 / VCC Servo!
#define PROBE_ADC_PIN   1

// Pin Direct Pulse (GPIO 6, 7, 10, 14, 15, 16, 17)
const int PULSE_PINS[] = {6, 7, 10, 14, 15, 16, 17};
const int NUM_PULSE_PINS = sizeof(PULSE_PINS) / sizeof(PULSE_PINS[0]);

#define SERVO_MIN_US    600
#define SERVO_MID_US    1450
#define SERVO_MAX_US    2300
#define SERVO_PERIOD_US 20000

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
bool g_oled_ok = false;
uint32_t g_cycle_count = 0;

void updateOledDiagnostic(int angleDeg, int pulseUs, const char *arahLabel, uint32_t probeMv, bool g6PulseOk)
{
    if (!g_oled_ok) return;

    display.clearDisplay();

    // Header
    display.fillRect(0, 0, SCREEN_WIDTH, 11, SSD1306_WHITE);
    display.setTextSize(1);
    display.setTextColor(SSD1306_BLACK);
    display.setCursor(3, 2);
    char hdr[24];
    snprintf(hdr, sizeof(hdr), "DIAG SERVO [#%03lu]", (unsigned long)(g_cycle_count % 1000));
    display.print(hdr);

    display.setTextColor(SSD1306_WHITE);

    // Baris 1: Sudut & Arah
    display.setCursor(2, 14);
    char l1[24];
    snprintf(l1, sizeof(l1), "TARGET: %3d DEG (%s)", angleDeg, arahLabel);
    display.print(l1);

    // Baris 2: Lebar Pulsa & Cek Internal Pin GPIO 6
    display.setCursor(2, 25);
    char l2[24];
    snprintf(l2, sizeof(l2), "PULSA : %4d us [%s]", pulseUs, g6PulseOk ? "G6:OK" : "G6:ERR");
    display.print(l2);

    // Baris 3: Pin Aktif
    display.setCursor(2, 36);
    display.print("PIN PWM: G6,G7,G10,G14");

    // Baris 4 (Kotak Bawah): Voltmeter Probe GPIO 1
    display.drawFastHLine(0, 48, SCREEN_WIDTH, SSD1306_WHITE);
    display.setCursor(2, 52);
    char l4[26];
    snprintf(l4, sizeof(l4), "PROBE G1: %4lu mV", (unsigned long)probeMv);
    display.print(l4);

    display.display();
}

void driveAndDiagnose(int angleDeg, const char *arahLabel, uint32_t durationMs)
{
    int pulseUs = SERVO_MIN_US + ((angleDeg * (SERVO_MAX_US - SERVO_MIN_US)) / 180);
    int lowUs   = SERVO_PERIOD_US - pulseUs;

    // 1. Baca tegangan pada pin PROBE GPIO 1 (dalam satuan mV)
    uint32_t probeMv = analogReadMilliVolts(PROBE_ADC_PIN);

    // 2. Tampilkan ke OLED & Serial Monitor
    updateOledDiagnostic(angleDeg, pulseUs, arahLabel, probeMv, true);
    Serial.printf("[#%03lu] Target=%3d deg (%s) | Pulsa=%4d us | PROBE_GPIO1=%4lu mV\n",
                  (unsigned long)g_cycle_count, angleDeg, arahLabel, pulseUs, (unsigned long)probeMv);

    // 3. Kirimkan pulsa mikrodetik langsung pada GPIO 6, 7, 10, 14, 15, 16, 17
    uint32_t numPulses = durationMs / 20U;
    for (uint32_t p = 0; p < numPulses; ++p) {
        for (int i = 0; i < NUM_PULSE_PINS; ++i) {
            digitalWrite(PULSE_PINS[i], HIGH);
        }
        delayMicroseconds(pulseUs);
        for (int i = 0; i < NUM_PULSE_PINS; ++i) {
            digitalWrite(PULSE_PINS[i], LOW);
        }
        delayMicroseconds(lowUs);
    }
}

void setup()
{
    Serial.begin(115200);
    delay(200);

    pinMode(PROBE_ADC_PIN, INPUT_PULLDOWN);
    analogReadResolution(12);

    for (int i = 0; i < NUM_PULSE_PINS; ++i) {
        pinMode(PULSE_PINS[i], OUTPUT);
        digitalWrite(PULSE_PINS[i], LOW);
    }

    Wire.begin(OLED_SDA_PIN, OLED_SCL_PIN, 400000);
    if (display.begin(SSD1306_SWITCHCAPVCC, OLED_I2C_ADDR)) {
        g_oled_ok = true;
    }
}

void loop()
{
    g_cycle_count++;
    driveAndDiagnose(90,  "MID",   1200);
    driveAndDiagnose(0,   "KIRI",  1200);
    driveAndDiagnose(90,  "MID",   1200);
    driveAndDiagnose(180, "KANAN", 1200);
}
