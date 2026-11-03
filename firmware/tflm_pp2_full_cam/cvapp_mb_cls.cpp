/*
 * cvapp_mb_cls.cpp
 *
 * Firmware PP2 Full Cam: Real-Time Input Kamera CIS (OV5647/HM0360)
 * On-Device MobileNetV2 (alpha=0.35) INT8 [1, 160, 160, 3] on Arm Ethos-U55 NPU
 * Transmisi Hasil AI:
 *   1. Paket I2C 5-Byte ke Arduino Uno R3 (Slave 0x55) via Pin PA2 (SCL) & PA3 (SDA)
 *   2. Streaming UART: String JSON Base64 JPEG + Hasil Prediksi untuk Live Viewer Python
 */

#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <forward_list>
#include <string>

#include "WE2_device.h"
#include "board.h"
#include "cvapp_mb_cls.h"
#include "cisdp_sensor.h"
#include "WE2_core.h"
#include "img_proc_helium.h"

extern "C" {
#include "hx_drv_scu.h"
#include "hx_drv_iic.h"
#include "hx_drv_uart.h"
#include "xprintf.h"
#include "memory_manage.h"
}

#include "ethosu_driver.h"
#include "pmu_ethosu.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#if TFLM2209_U55TAG2205
#include "tensorflow/lite/micro/micro_error_reporter.h"
#endif

#include "send_result.h"

using namespace std;

#define CPU_CLK              (0xffffff + 1)
#define PP2_INPUT_WIDTH      160
#define PP2_INPUT_HEIGHT     160
#define PP2_IMAGE_BYTES      (PP2_INPUT_WIDTH * PP2_INPUT_HEIGHT * 3)
#define ARDUINO_I2C_SLAVE_ADDR (0x55)

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

struct ethosu_driver ethosu_drv;
tflite::MicroInterpreter *mb_cls_int_ptr = nullptr;
TfLiteTensor *mb_cls_input = nullptr;
TfLiteTensor *mb_cls_output = nullptr;

static uint32_t capture_image_tick = 0;
static uint32_t frame_counter = 0;

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

static void i2cm_0_err_cb(void *status)
{
    (void)status;
}

static void init_i2c_master_for_arduino(void)
{
    hx_drv_scu_set_PA2_pinmux(SCU_PA2_PINMUX_I2C_M_SCL, 1);
    hx_drv_scu_set_PA3_pinmux(SCU_PA3_PINMUX_I2C_M_SDA, 1);
    hx_drv_i2cm_init(USE_DW_IIC_0, HX_I2C_HOST_MST_0_BASE, DW_IIC_SPEED_STANDARD);
    hx_drv_i2cm_set_err_cb(USE_DW_IIC_0, (void *)i2cm_0_err_cb);
}

static const char *send_i2c_packet_safe(uint8_t slave_addr, const uint8_t *pkt, uint32_t len)
{
    volatile uint32_t *i2c_regs = (volatile uint32_t *)HX_I2C_HOST_MST_0_BASE;

    i2c_regs[12] = 0; // Disable interrupts

    if ((i2c_regs[1] & 0x3FFu) != slave_addr) {
        i2c_regs[27] = 0; // IC_ENABLE = 0
        i2c_regs[1]  = (uint32_t)slave_addr;
        i2c_regs[27] = 1; // IC_ENABLE = 1
    } else if ((i2c_regs[27] & 0x01u) == 0u) {
        i2c_regs[27] = 1;
    }

    (void)i2c_regs[21];
    (void)i2c_regs[24];

    if ((i2c_regs[28] & 0x04u) == 0u) {
        i2c_regs[27] = 0;
        i2c_regs[1]  = (uint32_t)slave_addr;
        i2c_regs[27] = 1;
    }

    for (uint32_t i = 0; i < len; ++i) {
        uint32_t cmd = (uint32_t)pkt[i];
        if (i + 1 == len) {
            cmd |= 0x200u; // Bit 9 = STOP
        }
        i2c_regs[4] = cmd;
    }

    const uint32_t max_i2c_polls = 400000;
    for (uint32_t poll = 0; poll < max_i2c_polls; ++poll) {
        uint32_t raw_stat = i2c_regs[13];
        if ((raw_stat & 0x40u) != 0u) {   // TX_ABRT
            (void)i2c_regs[21];
            return "NO_ACK";
        }
        if ((raw_stat & 0x200u) != 0u) {  // STOP_DET
            uint32_t st = i2c_regs[28];
            if ((st & 0x04u) != 0u && (st & 0x20u) == 0u) {
                (void)i2c_regs[24];
                return "ACK_OK";
            }
        }
    }

    i2c_regs[27] = 0;
    i2c_regs[27] = 1;
    return "NO_ACK";
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
    xprintf("INFO -   PP2 FULL CAM: REAL-TIME ON-DEVICE CAMERA INFERENCE & ARDUINO I2C (Grove Vision AI V2)   \r\n");
    xprintf("INFO - =========================================================================================\r\n");
    xprintf("INFO - Target Device    : Seeed Grove Vision AI Module V2 (Cortex-M55 @ 400MHz + Ethos-U55 NPU)\r\n");
    xprintf("INFO - Model Arsitektur : MobileNetV2 (alpha=0.35) INT8 [1, 160, 160, 3] | Flash: 0x%08x\r\n", (unsigned int)model_addr);

    init_i2c_master_for_arduino();
    xprintf("INFO - I2C Master Bus   : PA2 (SCL) & PA3 (SDA) @ 100kHz -> Target Arduino Slave (0x%02X)\r\n",
            ARDUINO_I2C_SLAVE_ADDR);

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
    xprintf("INFO - Input Tensor     : [%d, %d, %d, %d] int8 (%d B) | Scale: %s | ZP: %d\r\n",
            (int)mb_cls_input->dims->data[0],
            (int)mb_cls_input->dims->data[1],
            (int)mb_cls_input->dims->data[2],
            (int)mb_cls_input->dims->data[3],
            (int)mb_cls_input->bytes,
            in_scale_str,
            (int)mb_cls_input->params.zero_point);
    xprintf("INFO - Output Tensor    : [%d, %d] int8 (%d B) | Scale: %s | ZP: %d\r\n",
            (int)mb_cls_output->dims->data[0],
            (int)mb_cls_output->dims->data[1],
            (int)mb_cls_output->bytes,
            out_scale_str,
            (int)mb_cls_output->params.zero_point);
    xprintf("INFO - Aturan Keputusan : Confidence >= 80.0%% & Pixel Std >= 12.0\r\n");
    xprintf("INFO - Paket I2C        : 5-Byte [0xAA, class_id, conf_int, conf_dec, checksum] -> Slave 0x55\r\n");
    xprintf("INFO - =========================================================================================\r\n");

    return 0;
}

int cv_mb_cls_run(struct_yolov8_ob_algoResult *algoresult_yolov8n_ob)
{
    (void)algoresult_yolov8n_ob;

    if (mb_cls_int_ptr == nullptr || mb_cls_input == nullptr || mb_cls_output == nullptr) {
        xprintf("INFO - [ERROR] Interpreter not initialized!\r\n");
        return -1;
    }

    uint32_t systick_1, systick_2;
    uint32_t loop_cnt_1, loop_cnt_2;
    SystemGetTick(&systick_1, &loop_cnt_1);

    // 1. Ambil citra mentah dari sensor kamera on-board (OV5647 / HM0360)
    uint32_t img_w = app_get_raw_width();
    uint32_t img_h = app_get_raw_height();
    uint32_t ch    = app_get_raw_channels();
    uint32_t raw_addr = app_get_raw_addr();

    if (raw_addr == 0 || img_w == 0 || img_h == 0) {
        sensordplib_retrigger_capture();
        return 0;
    }

    // 2. Hardware-accelerated image resize BGR8U3C -> RGB24 160x160 (Helium Vector MVE)
    float w_scale = (float)(img_w - 1) / (float)(PP2_INPUT_WIDTH - 1);
    float h_scale = (float)(img_h - 1) / (float)(PP2_INPUT_HEIGHT - 1);

    hx_lib_image_resize_BGR8U3C_to_RGB24_helium(
        (uint8_t *)raw_addr,
        (uint8_t *)mb_cls_input->data.uint8,
        img_w, img_h, ch,
        PP2_INPUT_WIDTH, PP2_INPUT_HEIGHT,
        w_scale, h_scale);

    // 3. Hitung Pixel Std Dev (deteksi frame kosong/tertutup) & Kuantisasi uint8 -> int8
    uint8_t *raw_rgb = mb_cls_input->data.uint8;
    int8_t *quant_buf = mb_cls_input->data.int8;

    uint32_t sum_bytes = 0;
    int64_t sum_sq = 0;

    for (size_t i = 0; i < PP2_IMAGE_BYTES; ++i) {
        uint32_t px = (uint32_t)raw_rgb[i];
        sum_bytes += px;
        sum_sq += (int64_t)(px * px);
        quant_buf[i] = (int8_t)((int32_t)px - 128); // Kuantisasi uint8 [0..255] -> int8 [-128..127]
    }

    float mean = (float)sum_bytes / (float)PP2_IMAGE_BYTES;
    float mean_sq = (float)sum_sq / (float)PP2_IMAGE_BYTES;
    float var = mean_sq - (mean * mean);
    float pixel_std = (var > 0.0f) ? sqrtf(var) : 0.0f;

    // Pastikan D-Cache diselaraskan sebelum DMA NPU membaca tensor input
    SCB_CleanDCache_by_Addr((uint32_t *)quant_buf, PP2_IMAGE_BYTES);

    // 4. Jalankan Inferensi di Arm Ethos-U55 NPU
    ETHOSU_PMU_CYCCNT_Reset(&ethosu_drv);
    ETHOSU_PMU_EVCNTR_ALL_Reset(&ethosu_drv);

    TfLiteStatus invoke_status = mb_cls_int_ptr->Invoke();
    if (invoke_status != kTfLiteOk) {
        xprintf("INFO - [ERROR] NPU Invoke() failed!\r\n");
        sensordplib_retrigger_capture();
        return -1;
    }

    uint32_t npu_total_cycles  = ETHOSU_PMU_Get_CCNTR(&ethosu_drv);
    float npu_ms  = (float)npu_total_cycles / 400000.0f; // 400 MHz core
    float npu_fps = (npu_ms > 0.0f) ? (1000.0f / npu_ms) : 0.0f;

    // 5. Dekuantisasi output [1, 2] INT8 -> Probabilitas Softmax
    const float out_scale = mb_cls_output->params.scale;
    const int32_t out_zp  = mb_cls_output->params.zero_point;
    int8_t raw_c0 = mb_cls_output->data.int8[0];
    int8_t raw_c1 = mb_cls_output->data.int8[1];

    float prob_c0 = ((float)raw_c0 - (float)out_zp) * out_scale;
    float prob_c1 = ((float)raw_c1 - (float)out_zp) * out_scale;
    if (prob_c0 < 0.0f) prob_c0 = 0.0f;
    if (prob_c1 < 0.0f) prob_c1 = 0.0f;
    float prob_sum = prob_c0 + prob_c1;
    if (prob_sum > 0.0f) {
        prob_c0 /= prob_sum;
        prob_c1 /= prob_sum;
    }

    float p_betina_pct = prob_c0 * 100.0f;
    float p_jantan_pct = prob_c1 * 100.0f;
    float max_conf_pct = (p_betina_pct >= p_jantan_pct) ? p_betina_pct : p_jantan_pct;

    // 6. Logika Keputusan Klasifikasi AI (Persis Algoritma v1)
    constexpr float CONFIDENCE_THRESHOLD = 80.0f;
    constexpr float BLANK_STD_THRESHOLD  = 12.0f;

    uint8_t class_id = 2;
    float reported_conf_pct = 0.0f;
    const char *label_str = "undefined";
    const char *reason_str = "OK";

    if (pixel_std < BLANK_STD_THRESHOLD) {
        class_id = 2;
        reported_conf_pct = 0.0f; // 0.00% -> Frame Kosong / Blank
        label_str = "undefined";
        reason_str = "BLANK_FRAME";
    } else if (max_conf_pct < CONFIDENCE_THRESHOLD) {
        class_id = 2;
        reported_conf_pct = max_conf_pct; // Ragu-ragu (< 80%)
        label_str = "undefined";
        reason_str = "LOW_CONF";
    } else if (p_betina_pct >= p_jantan_pct) {
        class_id = 0;
        reported_conf_pct = p_betina_pct;
        label_str = "Betina (hen)";
        reason_str = "OK";
    } else {
        class_id = 1;
        reported_conf_pct = p_jantan_pct;
        label_str = "Jantan (cock)";
        reason_str = "OK";
    }

    // 7. Format Paket I2C 5-Byte ke Arduino Uno R3 (Slave 0x55)
    int32_t conf_scaled = (int32_t)(reported_conf_pct * 100.0f + 0.5f);
    if (conf_scaled < 0) conf_scaled = 0;
    if (conf_scaled > 10000) conf_scaled = 10000;
    uint8_t conf_int = (uint8_t)(conf_scaled / 100);
    uint8_t conf_dec = (uint8_t)(conf_scaled % 100);

    uint8_t i2c_pkt[5];
    i2c_pkt[0] = 0xAA;
    i2c_pkt[1] = class_id;
    i2c_pkt[2] = conf_int;
    i2c_pkt[3] = conf_dec;
    i2c_pkt[4] = (uint8_t)(i2c_pkt[0] ^ i2c_pkt[1] ^ i2c_pkt[2] ^ i2c_pkt[3]);

    const char *i2c_status_str = send_i2c_packet_safe(ARDUINO_I2C_SLAVE_ADDR, i2c_pkt, 5);

    frame_counter++;

    // 8. Streaming UART: Output Base64 JPEG Image + JSON Prediksi untuk Python Viewer
    char conf_s[16], pb_s[16], pj_s[16], std_s[16], ms_s[16], fps_s[16];
    format_float(conf_s, sizeof(conf_s), reported_conf_pct, 2);
    format_float(pb_s, sizeof(pb_s), p_betina_pct, 2);
    format_float(pj_s, sizeof(pj_s), p_jantan_pct, 2);
    format_float(std_s, sizeof(std_s), pixel_std, 2);
    format_float(ms_s, sizeof(ms_s), npu_ms, 2);
    format_float(fps_s, sizeof(fps_s), npu_fps, 1);

    // Kirim string JSON SSCMA dengan base64 JPEG gambar kamera
    uint32_t jpeg_sz = app_get_jpeg_sz();
    uint32_t jpeg_addr = app_get_jpeg_addr();

    if (jpeg_sz > 0 && jpeg_addr != 0) {
        hx_InvalidateDCache_by_Addr((volatile void *)jpeg_addr, sizeof(uint8_t) * jpeg_sz);

        el_img_t live_jpg_img = el_img_t{};
        live_jpg_img.data   = (uint8_t *)jpeg_addr;
        live_jpg_img.size   = jpeg_sz;
        live_jpg_img.width  = img_w;
        live_jpg_img.height = img_h;
        live_jpg_img.format = EL_PIXEL_FORMAT_JPEG;
        live_jpg_img.rotate = EL_PIXEL_ROTATE_0;

        // Cetak JSON terintegrasi yang mudah dibaca oleh Python Viewer
        std::string payload_json = concat_strings(
            ", \"frame\": ", std::to_string(frame_counter),
            ", \"class_id\": ", std::to_string(class_id),
            ", \"label\": \"", label_str, "\"",
            ", \"confidence\": ", conf_s,
            ", \"p_betina\": ", pb_s,
            ", \"p_jantan\": ", pj_s,
            ", \"pixel_std\": ", std_s,
            ", \"reason\": \"", reason_str, "\"",
            ", \"npu_ms\": ", ms_s,
            ", \"npu_fps\": ", fps_s,
            ", \"i2c_status\": \"", i2c_status_str, "\"",
            ", ", img_2_json_str(&live_jpg_img)
        );

        event_reply(payload_json);
    } else {
        // Fallback cetak format PP2_JSON jika JPEG hardware encoder tidak aktif
        xprintf("PP2_JSON:{\"frame\":%u,\"class_id\":%d,\"label\":\"%s\",\"conf\":%s,\"p_betina\":%s,\"p_jantan\":%s,"
                "\"pixel_std\":%s,\"reason\":\"%s\",\"i2c_status\":\"%s\",\"npu_ms\":%s,\"npu_fps\":%s}\r\n",
                (unsigned int)frame_counter, (int)class_id, label_str, conf_s, pb_s, pj_s,
                std_s, reason_str, i2c_status_str, ms_s, fps_s);
    }

    SystemGetTick(&systick_2, &loop_cnt_2);
    capture_image_tick = (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2);

    // 9. Re-trigger capture frame berikutnya dari sensor CIS
    sensordplib_retrigger_capture();

    return 0;
}

int cv_mb_cls_deinit()
{
    return 0;
}
