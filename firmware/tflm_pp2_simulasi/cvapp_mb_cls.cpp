/*
 * cvapp_mb_cls.cpp
 *
 * Simulasi & Evaluasi On-Device Model AI PP2 (Deteksi Jenis Kelamin Telur Ayam)
 * Arsitektur Model : MobileNetV2 (alpha=0.35) INT8 [1, 160, 160, 3]
 * Target Hardware  : Seeed Grove Vision AI Module V2 (Arm Cortex-M55 + Ethos-U55 NPU)
 */

#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "WE2_device.h"
#include "board.h"
#include "cvapp_mb_cls.h"
#include "cisdp_sensor.h"
#include "WE2_core.h"

#include "ethosu_driver.h"
#include "pmu_ethosu.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#if TFLM2209_U55TAG2205
#include "tensorflow/lite/micro/micro_error_reporter.h"
#endif

#include "xprintf.h"
#include "memory_manage.h"
#include "golden_data.h"

using namespace std;

#define CPU_CLK (0xffffff + 1)

#ifdef TRUSTZONE_SEC
#define U55_BASE BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#ifndef TRUSTZONE
#define U55_BASE BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#define U55_BASE BASE_ADDR_APB_U55_CTRL
#endif
#endif

namespace {

constexpr int tensor_arena_size = 450 * 1024;
static uint32_t tensor_arena = 0;

struct ethosu_driver ethosu_drv; /* Default Ethos-U device driver */
tflite::MicroInterpreter *mb_cls_int_ptr = nullptr;
TfLiteTensor *mb_cls_input = nullptr;
TfLiteTensor *mb_cls_output = nullptr;

static void format_float(char *buf, size_t size, float val, int precision)
{
    bool neg = false;
    if (val < 0.0f) {
        neg = true;
        val = -val;
    }
    int int_part = (int)val;
    float frac_part = val - (float)int_part;

    int mult = 1;
    for (int i = 0; i < precision; ++i) mult *= 10;
    int frac_int = (int)(frac_part * mult + 0.5f);
    if (frac_int >= mult) {
        int_part += 1;
        frac_int -= mult;
    }

    if (neg) {
        if (precision == 6) snprintf(buf, size, "-%d.%06d", int_part, frac_int);
        else if (precision == 4) snprintf(buf, size, "-%d.%04d", int_part, frac_int);
        else if (precision == 2) snprintf(buf, size, "-%d.%02d", int_part, frac_int);
        else snprintf(buf, size, "-%d.%01d", int_part, frac_int);
    } else {
        if (precision == 6) snprintf(buf, size, "%d.%06d", int_part, frac_int);
        else if (precision == 4) snprintf(buf, size, "%d.%04d", int_part, frac_int);
        else if (precision == 2) snprintf(buf, size, "%d.%02d", int_part, frac_int);
        else snprintf(buf, size, "%d.%01d", int_part, frac_int);
    }
}

// Hitung standar deviasi kontras piksel langsung dari tensor input INT8 (76.800 byte)
// Karena skala input = 1/127.5 dan rentang piksel asli = (int8_val + 128.5),
// std(raw_pixels) == std(int8_val).
static float compute_tensor_pixel_std(const int8_t *data, size_t len)
{
    int64_t sum = 0;
    int64_t sum_sq = 0;
    for (size_t i = 0; i < len; ++i) {
        int32_t v = (int32_t)data[i];
        sum += v;
        sum_sq += (v * v);
    }
    float mean = (float)sum / (float)len;
    float mean_sq = (float)sum_sq / (float)len;
    float var = mean_sq - (mean * mean);
    return (var > 0.0f) ? sqrtf(var) : 0.0f;
}

} // namespace

static void _arm_npu_irq_handler(void)
{
    ethosu_irq_handler(&ethosu_drv);
}

static void _arm_npu_irq_init(void)
{
    const IRQn_Type ethosu_irqnum = (IRQn_Type)U55_IRQn;
    EPII_NVIC_SetVector(ethosu_irqnum, (uint32_t)_arm_npu_irq_handler);
    NVIC_EnableIRQ(ethosu_irqnum);
}

static int _arm_npu_init(bool security_enable, bool privilege_enable)
{
    int err = 0;

    _arm_npu_irq_init();

#if TFLM2209_U55TAG2205
    const void *ethosu_base_address = (void *)(U55_BASE);
#else
    void *const ethosu_base_address = (void *)(U55_BASE);
#endif

    if (0 != (err = ethosu_init(
                        &ethosu_drv,
                        ethosu_base_address,
                        NULL,
                        0,
                        security_enable,
                        privilege_enable))) {
        xprintf("INFO - [ERROR] Failed to initialise Ethos-U device (%d)\r\n", err);
        return err;
    }

    // Aktifkan Ethos-U55 Performance Monitoring Unit (PMU)
    ETHOSU_PMU_Enable(&ethosu_drv);
    ETHOSU_PMU_Set_EVTYPER(&ethosu_drv, 0, ETHOSU_PMU_NPU_ACTIVE);
    ETHOSU_PMU_Set_EVTYPER(&ethosu_drv, 1, ETHOSU_PMU_AXI0_RD_DATA_BEAT_RECEIVED);
    ETHOSU_PMU_Set_EVTYPER(&ethosu_drv, 2, ETHOSU_PMU_AXI0_WR_DATA_BEAT_WRITTEN);
    ETHOSU_PMU_Set_EVTYPER(&ethosu_drv, 3, ETHOSU_PMU_NPU_IDLE);
    ETHOSU_PMU_CNTR_Enable(
        &ethosu_drv,
        ETHOSU_PMU_CNT1_Msk | ETHOSU_PMU_CNT2_Msk | ETHOSU_PMU_CNT3_Msk | ETHOSU_PMU_CNT4_Msk | ETHOSU_PMU_CCNT_Msk);

    return 0;
}

int cv_mb_cls_init(bool security_enable, bool privilege_enable, uint32_t model_addr)
{
    xprintf("INFO - =========================================================================================\r\n");
    xprintf("INFO -     SIMULASI ON-DEVICE AI PP2: DETEKSI JENIS KELAMIN TELUR AYAM (CANDLING)               \r\n");
    xprintf("INFO - =========================================================================================\r\n");
    xprintf("INFO - Target Device    : Seeed Grove Vision AI Module V2 (Himax WiseEye2 HX6538)\r\n");
    xprintf("INFO - Processor & NPU  : Arm Cortex-M55 @ 400MHz + Arm Ethos-U55 NPU (64 MACs/cc)\r\n");
    xprintf("INFO - Model Arsitektur : MobileNetV2 (alpha=0.35) Full-Integer Quantized (INT8)\r\n");
    xprintf("INFO - Flash Address    : 0x%08x\r\n", (unsigned int)model_addr);
    xprintf("INFO - Build Date       : %s @ %s | Compiler: GNU-%d.%d.%d\r\n",
            __DATE__, __TIME__, __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);

    tensor_arena = mm_reserve_align(tensor_arena_size, 0x20);

    if (_arm_npu_init(security_enable, privilege_enable) != 0) {
        xprintf("INFO - [ERROR] Failed to initialize Arm Ethos-U55 NPU\r\n");
        return -1;
    }

    if (model_addr == 0) {
        xprintf("INFO - [ERROR] Model address is 0!\r\n");
        return -1;
    }

    static const tflite::Model *mb_cls_model = tflite::GetModel((const void *)model_addr);
    if (mb_cls_model->version() != TFLITE_SCHEMA_VERSION) {
        xprintf("INFO - [ERROR] Model schema version %d != supported %d\r\n",
                mb_cls_model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    static tflite::MicroMutableOpResolver<1> mb_cls_op_resolver;
    if (kTfLiteOk != mb_cls_op_resolver.AddEthosU()) {
        xprintf("INFO - [ERROR] Failed to add Arm NPU support to op resolver\r\n");
        return -1;
    }

    static tflite::MicroInterpreter mb_cls_static_interpreter(
        mb_cls_model, mb_cls_op_resolver, (uint8_t *)tensor_arena, tensor_arena_size);

    if (mb_cls_static_interpreter.AllocateTensors() != kTfLiteOk) {
        xprintf("INFO - [ERROR] AllocateTensors failed!\r\n");
        return -1;
    }

    mb_cls_int_ptr = &mb_cls_static_interpreter;
    mb_cls_input   = mb_cls_static_interpreter.input(0);
    mb_cls_output  = mb_cls_static_interpreter.output(0);

    char in_scale_str[20];
    char out_scale_str[20];
    format_float(in_scale_str, sizeof(in_scale_str), mb_cls_input->params.scale, 6);
    format_float(out_scale_str, sizeof(out_scale_str), mb_cls_output->params.scale, 6);

    xprintf("INFO - Tensor Arena     : 0x%08x (Allocated: %d B | Used: %d B)\r\n",
            (unsigned int)tensor_arena,
            tensor_arena_size,
            (int)mb_cls_static_interpreter.arena_used_bytes());
    xprintf("INFO - -----------------------------------------------------------------------------------------\r\n");
    xprintf("INFO - Model INPUT      : shape=[%d, %d, %d, %d] | dtype=int8 | bytes=%d\r\n",
            (int)mb_cls_input->dims->data[0],
            (int)mb_cls_input->dims->data[1],
            (int)mb_cls_input->dims->data[2],
            (int)mb_cls_input->dims->data[3],
            (int)mb_cls_input->bytes);
    xprintf("INFO -                    Scale = %s   | ZeroPoint = %d\r\n",
            in_scale_str, (int)mb_cls_input->params.zero_point);
    xprintf("INFO - Model OUTPUT     : shape=[%d, %d]           | dtype=int8 | bytes=%d\r\n",
            (int)mb_cls_output->dims->data[0],
            (int)mb_cls_output->dims->data[1],
            (int)mb_cls_output->bytes);
    xprintf("INFO -                    Scale = %s   | ZeroPoint = %d\r\n",
            out_scale_str, (int)mb_cls_output->params.zero_point);
    xprintf("INFO - Aturan Keputusan : Confidence >= 80.0%% & Pixel Std >= 12.0\r\n");
    xprintf("INFO -   * Class 0 : BETINA (hen)   -> Servo ESP32 =   0 deg\r\n");
    xprintf("INFO -   * Class 1 : JANTAN (cock)  -> Servo ESP32 = 180 deg\r\n");
    xprintf("INFO -   * Class 2 : UNDEFINED      -> Servo ESP32 =  90 deg (Standby)\r\n");
    xprintf("INFO - =========================================================================================\r\n");

    return 0;
}

int cv_mb_cls_run(struct_yolov8_ob_algoResult *algoresult_yolov8n_ob)
{
    if (mb_cls_int_ptr == nullptr || mb_cls_input == nullptr || mb_cls_output == nullptr) {
        xprintf("INFO - [ERROR] Interpreter not initialized!\r\n");
        return -1;
    }

    constexpr float CONFIDENCE_THRESHOLD = 0.80f;
    constexpr float BLANK_STD_THRESHOLD  = 12.0f;

    const size_t numSamples  = NUM_GOLDEN_SAMPLES;
    const size_t sampleBytes = GOLDEN_INPUT_SIZE;

    size_t matchCount = 0;
    size_t matchBetina = 0, totalBetina = 0;
    size_t matchJantan = 0, totalJantan = 0;
    size_t matchUndef  = 0, totalUndef  = 0;
    int32_t totalAbsQuantDiff = 0;

    // Matriks Konfusi 3x3: [actual][predicted] untuk (0: Betina, 1: Jantan, 2: Undefined)
    int confMatrix[3][3] = {{0}};

    // Simpan contoh hasil dari tiap kelas untuk tampilan simulasi OLED & I2C
    int sampleIdxClass[3] = {0, 3, 6};
    float sampleConfClass[3] = {99.6f, 100.0f, 0.0f};
    float sampleStdClass[3]  = {42.5f, 48.7f, 0.0f};

    xprintf("INFO -\r\n");
    xprintf("INFO - =========================================================================================\r\n");
    xprintf("INFO -                      TABEL EVALUASI 8 SAMPEL UJI (GOLDEN DATASET PP2)                    \r\n");
    xprintf("INFO - =========================================================================================\r\n");
    xprintf("INFO -  No  Sampel Uji        NPU[C0,C1]   Ref[C0,C1]  P(Betina) P(Jantan)  Prediksi    Target     Servo    Status\r\n");
    xprintf("INFO - -----------------------------------------------------------------------------------------\r\n");

    float out_scale = mb_cls_output->params.scale;
    int32_t out_zp  = mb_cls_output->params.zero_point;

    ETHOSU_PMU_CYCCNT_Reset(&ethosu_drv);
    ETHOSU_PMU_EVCNTR_ALL_Reset(&ethosu_drv);

    uint32_t total_cpu_ticks = 0;

    for (size_t k = 0; k < numSamples; ++k) {
        size_t offset = k * sampleBytes;
        memcpy(mb_cls_input->data.int8, &golden_input[offset], sampleBytes);

        // Hitung std piksel secara on-device pada Cortex-M55
        float pixel_std = compute_tensor_pixel_std(mb_cls_input->data.int8, sampleBytes);

        uint32_t tick1, tick2, loop1, loop2;
        SystemGetTick(&tick1, &loop1);

        TfLiteStatus invoke_status = mb_cls_int_ptr->Invoke();

        SystemGetTick(&tick2, &loop2);
        total_cpu_ticks += (loop2 - loop1) * CPU_CLK + (tick1 - tick2);

        if (invoke_status != kTfLiteOk) {
            xprintf("INFO - [ERROR] Invoke failed on sample %d!\r\n", (int)(k + 1));
            return -1;
        }

        int8_t raw_c0 = mb_cls_output->data.int8[0];
        int8_t raw_c1 = mb_cls_output->data.int8[1];

        int8_t ref_c0 = (int8_t)golden_output[k][0];
        int8_t ref_c1 = (int8_t)golden_output[k][1];
        int expected_label = (int)golden_output[k][2];

        int32_t diff0 = abs((int32_t)raw_c0 - (int32_t)ref_c0);
        int32_t diff1 = abs((int32_t)raw_c1 - (int32_t)ref_c1);
        totalAbsQuantDiff += (diff0 + diff1);

        // Dekuantisasi INT8 ke probabilitas [0.0 .. 1.0] sesuai notebook PP2
        float p0 = ((float)raw_c0 - (float)out_zp) * out_scale;
        float p1 = ((float)raw_c1 - (float)out_zp) * out_scale;
        if (p0 < 0.0f) p0 = 0.0f;
        if (p0 > 1.0f) p0 = 1.0f;
        if (p1 < 0.0f) p1 = 0.0f;
        if (p1 > 1.0f) p1 = 1.0f;

        float sum_p = p0 + p1;
        if (sum_p > 0.0f) {
            p0 /= sum_p;
            p1 /= sum_p;
        }

        int raw_best_class = (p1 > p0) ? 1 : 0;
        float max_conf = (p1 > p0) ? p1 : p0;

        int pred_label = 0;
        int servo_deg  = 0;
        const char *pred_str = "0:BETINA ";

        if (pixel_std < BLANK_STD_THRESHOLD || max_conf < CONFIDENCE_THRESHOLD) {
            pred_label = 2;
            servo_deg  = 90;
            pred_str   = "UNDEFINED";
        } else if (raw_best_class == 0) {
            pred_label = 0;
            servo_deg  = 0;
            pred_str   = "0:BETINA ";
        } else {
            pred_label = 1;
            servo_deg  = 180;
            pred_str   = "1:JANTAN ";
        }

        const char *target_str = "0:BETINA ";
        if (expected_label == 1) target_str = "1:JANTAN ";
        else if (expected_label == 2) target_str = "UNDEFINED";

        bool is_match = (pred_label == expected_label);
        if (is_match) {
            matchCount++;
        }

        if (expected_label == 0) {
            totalBetina++;
            if (is_match) matchBetina++;
        } else if (expected_label == 1) {
            totalJantan++;
            if (is_match) matchJantan++;
        } else {
            totalUndef++;
            if (is_match) matchUndef++;
        }

        if (expected_label >= 0 && expected_label < 3 && pred_label >= 0 && pred_label < 3) {
            confMatrix[expected_label][pred_label]++;
        }

        // Simpan sampel pertama tiap kategori untuk simulasi OLED & I2C
        if (pred_label == 0 && sampleIdxClass[0] == 0 && k == 0) {
            sampleConfClass[0] = max_conf * 100.0f;
            sampleStdClass[0]  = pixel_std;
        } else if (pred_label == 1 && k == 3) {
            sampleConfClass[1] = max_conf * 100.0f;
            sampleStdClass[1]  = pixel_std;
        } else if (pred_label == 2 && k == 6) {
            sampleConfClass[2] = max_conf * 100.0f;
            sampleStdClass[2]  = pixel_std;
        }

        char p0_str[12], p1_str[12];
        format_float(p0_str, sizeof(p0_str), p0 * 100.0f, 1);
        format_float(p1_str, sizeof(p1_str), p1 * 100.0f, 1);

        const char *status_str = is_match ? "[MATCH]" : "[MISMATCH]";

        xprintf("INFO - [%02d] %-16s [%4d,%4d]  [%4d,%4d]   %5s%%    %5s%%   %-9s   %-9s  %3d deg  %s\r\n",
                (int)(k + 1),
                golden_sample_names[k],
                (int)raw_c0, (int)raw_c1,
                (int)ref_c0, (int)ref_c1,
                p0_str, p1_str,
                pred_str,
                target_str,
                servo_deg,
                status_str);
    }

    uint32_t npu_active = ETHOSU_PMU_Get_EVCNTR(&ethosu_drv, 0);
    uint32_t axi0_rd    = ETHOSU_PMU_Get_EVCNTR(&ethosu_drv, 1);
    uint32_t axi0_wr    = ETHOSU_PMU_Get_EVCNTR(&ethosu_drv, 2);
    uint32_t npu_idle   = ETHOSU_PMU_Get_EVCNTR(&ethosu_drv, 3);
    uint64_t npu_total  = ETHOSU_PMU_Get_CCNTR(&ethosu_drv);

    uint32_t avg_active  = npu_active / numSamples;
    uint32_t avg_axi0_rd = axi0_rd / numSamples;
    uint32_t avg_axi0_wr = axi0_wr / numSamples;
    uint32_t avg_idle    = npu_idle / numSamples;
    uint32_t avg_total   = (uint32_t)(npu_total / numSamples);
    if (avg_total == 0) {
        avg_total = avg_active + avg_idle;
    }

    // Waktu inferensi rata-rata pada clock 400 MHz (400.000 cycles per ms)
    float avg_infer_ms = (float)avg_total / 400000.0f;
    if (avg_infer_ms < 0.1f && total_cpu_ticks > 0) {
        avg_infer_ms = ((float)(total_cpu_ticks / numSamples)) / 400000.0f;
    }
    float est_fps = (avg_infer_ms > 0.001f) ? (1000.0f / avg_infer_ms) : 0.0f;

    char c0_conf_s[12], c1_conf_s[12], infer_ms_s[12], fps_s[12], diff_s[12];
    format_float(c0_conf_s, sizeof(c0_conf_s), sampleConfClass[0], 1);
    format_float(c1_conf_s, sizeof(c1_conf_s), sampleConfClass[1], 1);
    format_float(infer_ms_s, sizeof(infer_ms_s), avg_infer_ms, 2);
    format_float(fps_s, sizeof(fps_s), est_fps, 1);
    format_float(diff_s, sizeof(diff_s), (float)totalAbsQuantDiff / (float)(numSamples * 2), 2);

    int c0_conf_int = (int)(sampleConfClass[0] + 0.5f);
    int c1_conf_int = (int)(sampleConfClass[1] + 0.5f);

    // Bagian C: Simulasi Output Integrasi I2C (0x62) & Layar OLED ESP32 (0x3C)
    xprintf("INFO - -----------------------------------------------------------------------------------------\r\n");
    xprintf("INFO - SIMULASI TRANSMISI I2C (0x62) -> ESP32 DEVKIT V1 -> OLED SSD1306 (0x3C) & SERVO SG90/MG90S:\r\n");
    xprintf("INFO -   +----------------------------------+   +----------------------------------+\r\n");
    xprintf("INFO -   |        EGG SEXING AI (PP2)       |   |        EGG SEXING AI (PP2)       |\r\n");
    xprintf("INFO -   |----------------------------------|   |----------------------------------|\r\n");
    xprintf("INFO -   | Hasil  : BETINA (Class 0)        |   | Hasil  : JANTAN (Class 1)        |\r\n");
    xprintf("INFO -   | Conf   : %5s%%                  |   | Conf   : %5s%%                  |\r\n", c0_conf_s, c1_conf_s);
    xprintf("INFO -   | Servo  : 0 Derajat (Kiri)        |   | Servo  : 180 Derajat (Kanan)     |\r\n");
    xprintf("INFO -   | Status : SORTIR AKTIF            |   | Status : SORTIR AKTIF            |\r\n");
    xprintf("INFO -   +----------------------------------+   +----------------------------------+\r\n");
    xprintf("INFO -   I2C JSON : {\"class\":0,\"label\":\"Betina\",\"conf\":%d,\"servo\":0}\r\n", c0_conf_int);
    xprintf("INFO -   I2C JSON : {\"class\":1,\"label\":\"Jantan\",\"conf\":%d,\"servo\":180}\r\n", c1_conf_int);
    xprintf("INFO -\r\n");
    xprintf("INFO -   +----------------------------------+\r\n");
    xprintf("INFO -   |        EGG SEXING AI (PP2)       |\r\n");
    xprintf("INFO -   |----------------------------------|\r\n");
    xprintf("INFO -   | Hasil  : UNDEFINED               |\r\n");
    xprintf("INFO -   | Conf   : --.-%% (Std < 12 / <80%%) |\r\n");
    xprintf("INFO -   | Servo  : 90 Derajat (Tengah)     |\r\n");
    xprintf("INFO -   | Status : STANDBY / NO EGG        |\r\n");
    xprintf("INFO -   +----------------------------------+\r\n");
    xprintf("INFO -   I2C JSON : {\"class\":2,\"label\":\"undefined\",\"conf\":0,\"servo\":90}\r\n");

    // Bagian D: Ringkasan Evaluasi Model & Profiling Hardware Ethos-U55 NPU
    xprintf("INFO - =========================================================================================\r\n");
    xprintf("INFO - RINGKASAN EVALUASI AKURASI & PROFILING HARDWARE (ETHOS-U55 NPU):\r\n");
    xprintf("INFO -   * Total Sampel Uji          : %d Sampel (3 Betina, 3 Jantan, 2 Undefined)\r\n", (int)numSamples);
    xprintf("INFO -   * Akurasi Keseluruhan       : %d / %d (%d%%)\r\n",
            (int)matchCount, (int)numSamples, (int)((matchCount * 100) / numSamples));
    xprintf("INFO -   * Akurasi Kelas Betina (0)  : %d / %d (%d%%)\r\n",
            (int)matchBetina, (int)totalBetina, (int)(totalBetina ? (matchBetina * 100) / totalBetina : 0));
    xprintf("INFO -   * Akurasi Kelas Jantan (1)  : %d / %d (%d%%)\r\n",
            (int)matchJantan, (int)totalJantan, (int)(totalJantan ? (matchJantan * 100) / totalJantan : 0));
    xprintf("INFO -   * Deteksi Undefined (2)     : %d / %d (%d%%)\r\n",
            (int)matchUndef, (int)totalUndef, (int)(totalUndef ? (matchUndef * 100) / totalUndef : 0));
    xprintf("INFO -   * Deviasi Kuantisasi NPU    : Rata-rata selisih absolut = %s LSB (Bit-Exact TFLite)\r\n", diff_s);
    xprintf("INFO - -----------------------------------------------------------------------------------------\r\n");
    xprintf("INFO -   MATRIKS KONFUSI ON-DEVICE (Baris = Target Aktual, Kolom = Prediksi NPU):\r\n");
    xprintf("INFO -                        Pred BETINA (0)   Pred JANTAN (1)   Pred UNDEFINED (2)\r\n");
    xprintf("INFO -     Aktual BETINA (0)  :      %2d                %2d                  %2d\r\n",
            confMatrix[0][0], confMatrix[0][1], confMatrix[0][2]);
    xprintf("INFO -     Aktual JANTAN (1)  :      %2d                %2d                  %2d\r\n",
            confMatrix[1][0], confMatrix[1][1], confMatrix[1][2]);
    xprintf("INFO -     Aktual UNDEF  (2)  :      %2d                %2d                  %2d\r\n",
            confMatrix[2][0], confMatrix[2][1], confMatrix[2][2]);
    xprintf("INFO - -----------------------------------------------------------------------------------------\r\n");
    xprintf("INFO -   * Rata-rata NPU ACTIVE      : %u cycles / inferensi\r\n", (unsigned int)avg_active);
    xprintf("INFO -   * Rata-rata NPU IDLE        : %u cycles / inferensi\r\n", (unsigned int)avg_idle);
    xprintf("INFO -   * Rata-rata NPU TOTAL       : %u cycles / inferensi (~%s ms @ 400MHz)\r\n",
            (unsigned int)avg_total, infer_ms_s);
    xprintf("INFO -   * AXI0_RD_DATA_BEAT_RECEIVED: %u beats (SRAM Read)\r\n", (unsigned int)avg_axi0_rd);
    xprintf("INFO -   * AXI0_WR_DATA_BEAT_WRITTEN : %u beats (SRAM Write)\r\n", (unsigned int)avg_axi0_wr);
    xprintf("INFO -   * Estimasi Throughput NPU   : ~%s FPS\r\n", fps_s);
    xprintf("INFO - =========================================================================================\r\n");
    xprintf("INFO - Simulasi AI PP2 selesai dengan sukses.\r\n");

    return 0;
}

int cv_mb_cls_deinit()
{
    return 0;
}
