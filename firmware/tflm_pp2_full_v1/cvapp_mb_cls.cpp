/*
 * cvapp_mb_cls.cpp
 *
 * Firmware PP2 Full v1: Mode Upload Gambar via USB Serial (Notebook 3)
 * & Transmisi Hasil AI (5-Byte I2C) ke ESP32 Master Controller
 *
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

using namespace std;

#define CPU_CLK              (0xffffff + 1)
#define PP2_IMAGE_BYTES      (160 * 160 * 3)
#define ESP32_I2C_SLAVE_ADDR (0x55)

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
    // Non-blocking error callback agar board tetap berjalan meskipun ESP32 belum terhubung
    (void)status;
}

static void init_i2c_master_for_esp32(void)
{
    hx_drv_scu_set_PA2_pinmux(SCU_PA2_PINMUX_I2C_M_SCL, 1);
    hx_drv_scu_set_PA3_pinmux(SCU_PA3_PINMUX_I2C_M_SDA, 1);
    hx_drv_i2cm_init(USE_DW_IIC_0, HX_I2C_HOST_MST_0_BASE, DW_IIC_SPEED_STANDARD);
    hx_drv_i2cm_set_err_cb(USE_DW_IIC_0, (void *)i2cm_0_err_cb);
}

static const char *send_i2c_packet_safe(uint8_t slave_addr, const uint8_t *pkt, uint32_t len)
{
    volatile uint32_t *i2c_regs = (volatile uint32_t *)HX_I2C_HOST_MST_0_BASE;

    // 1. Matikan seluruh interrupt I2C (IC_INTR_MASK = offset 0x30 -> index 12)
    //    agar ISR bawaan SDK (dw_iic_mst0_isr) tidak memotong pengiriman polling kita
    i2c_regs[12] = 0;

    // 2. Set target slave address (IC_TAR = offset 0x04 -> index 1) jika berbeda
    if ((i2c_regs[1] & 0x3FFu) != slave_addr) {
        i2c_regs[27] = 0; // IC_ENABLE = 0
        i2c_regs[1]  = (uint32_t)slave_addr;
        i2c_regs[27] = 1; // IC_ENABLE = 1
    } else if ((i2c_regs[27] & 0x01u) == 0u) {
        i2c_regs[27] = 1;
    }

    // 3. Bersihkan flag interrupt lama (IC_CLR_TX_ABRT = idx 21, IC_CLR_STOP_DET = idx 24)
    (void)i2c_regs[21];
    (void)i2c_regs[24];

    // 4. Pastikan TX FIFO kosong sebelum mulai; jika masih nyangkut dari frame sebelumnya, reset FIFO
    if ((i2c_regs[28] & 0x04u) == 0u) {
        i2c_regs[27] = 0;
        i2c_regs[1]  = (uint32_t)slave_addr;
        i2c_regs[27] = 1;
    }

    // 5. Masukkan 5 byte paket ke IC_DATA_CMD (offset 0x10 -> index 4), byte terakhir diberi bit STOP (0x200)
    for (uint32_t i = 0; i < len; ++i) {
        uint32_t cmd = (uint32_t)pkt[i];
        if (i + 1 == len) {
            cmd |= 0x200u; // Bit 9 = STOP
        }
        i2c_regs[4] = cmd;
    }

    // 6. Tunggu transmisi 5 byte @ 100kHz (~0.6 ms) selesai dengan batas timeout ~15 ms (400.000 pembacaan APB)
    const uint32_t max_i2c_polls = 400000;
    for (uint32_t poll = 0; poll < max_i2c_polls; ++poll) {
        uint32_t raw_stat = i2c_regs[13]; // IC_RAW_INTR_STAT (offset 0x34 -> index 13)
        if ((raw_stat & 0x40u) != 0u) {   // Bit 6 = TX_ABRT (Slave NACK)
            (void)i2c_regs[21];           // Clear TX_ABRT
            return "NO_ACK";
        }
        if ((raw_stat & 0x200u) != 0u) {  // Bit 9 = STOP_DET
            uint32_t st = i2c_regs[28];   // IC_STATUS (offset 0x70 -> index 28)
            if ((st & 0x04u) != 0u && (st & 0x20u) == 0u) { // TFE=1 & MST_ACTIVITY=0
                (void)i2c_regs[24];       // Clear STOP_DET
                return "ACK_OK";
            }
        }
    }

    // Timeout (misal jalur SCL/SDA tertahan LOW karena ESP32 mati/belum terhubung): reset TX FIFO
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
    xprintf("INFO -     PP2 FULL v1: DETEKSI JENIS KELAMIN TELUR AYAM (UPLOAD GAMBAR VIA USB SERIAL)         \r\n");
    xprintf("INFO - =========================================================================================\r\n");
    xprintf("INFO - Target Device    : Seeed Grove Vision AI Module V2 (Cortex-M55 @ 400MHz + Ethos-U55 NPU)\r\n");
    xprintf("INFO - Model Arsitektur : MobileNetV2 (alpha=0.35) INT8 | Flash Addr: 0x%08x\r\n", (unsigned int)model_addr);
    xprintf("INFO - Build Date       : %s @ %s | Compiler: GNU-%d.%d.%d\r\n",
            __DATE__, __TIME__, __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);

    init_i2c_master_for_esp32();
    xprintf("INFO - I2C Master Bus   : PA2 (SCL) & PA3 (SDA) @ 100kHz -> Target ESP32 Slave (0x%02X)\r\n",
            ESP32_I2C_SLAVE_ADDR);

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
    xprintf("INFO - Output Tensor    : [%d, %d]           int8 (%d B)     | Scale: %s | ZP: %d\r\n",
            (int)mb_cls_output->dims->data[0],
            (int)mb_cls_output->dims->data[1],
            (int)mb_cls_output->bytes,
            out_scale_str,
            (int)mb_cls_output->params.zero_point);
    xprintf("INFO - Aturan Keputusan : Confidence >= 80.0%% & Pixel Std >= 12.0\r\n");
    xprintf("INFO - Paket I2C ESP32  : 5-Byte [0xAA, class_id, conf_int, conf_dec, checksum] -> 0x55\r\n");
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

    DEV_UART *console_uart = hx_drv_uart_get_dev(USE_DW_UART_0);
    console_uart->uart_open(UART_BAUDRATE_921600);
    volatile uint32_t *uart_regs =
        (volatile uint32_t *)(((uint32_t *)console_uart->uart_info.uart_ctrl)[1]);

    constexpr float CONFIDENCE_THRESHOLD = 80.0f;
    constexpr float BLANK_STD_THRESHOLD  = 12.0f;
    const float out_scale = mb_cls_output->params.scale;
    const int32_t out_zp  = mb_cls_output->params.zero_point;

    uint32_t frame_counter = 0;
    uint8_t cmd_win[5] = {0, 0, 0, 0, 0};

    xprintf("INFO - [READY] Menunggu upload gambar 160x160 RGB dari Notebook 3 (Protokol: PP2IR + 76800 B)...\r\n");

    while (1) {
        if ((uart_regs[5] & 0x01u) == 0u && (uart_regs[31] & 0x08u) == 0u) {
            continue;
        }
        uint8_t rx_byte = (uint8_t)(uart_regs[0] & 0xFFu);

        // Geser sliding window 5 byte untuk mendeteksi header "PP2IR" atau "PP2IP"
        cmd_win[0] = cmd_win[1];
        cmd_win[1] = cmd_win[2];
        cmd_win[2] = cmd_win[3];
        cmd_win[3] = cmd_win[4];
        cmd_win[4] = rx_byte;

        // Periksa command PING: "PP2IP"
        if (cmd_win[0] == 'P' && cmd_win[1] == 'P' && cmd_win[2] == '2' &&
            cmd_win[3] == 'I' && cmd_win[4] == 'P') {
            memset(cmd_win, 0, sizeof(cmd_win));
            xprintf("PP2_PONG:READY\r\n");
            continue;
        }

        // Periksa command Upload Gambar RGB: "PP2IR"
        if (cmd_win[0] == 'P' && cmd_win[1] == 'P' && cmd_win[2] == '2' &&
            cmd_win[3] == 'I' && cmd_win[4] == 'R') {
            memset(cmd_win, 0, sizeof(cmd_win));

            // Kirim Handshake RDY ke Laptop
            const char rdy_msg[] = "RDY\r\n";
            console_uart->uart_write(rdy_msg, 5);

            // Terima 76.800 byte piksel RGB langsung dari register hardware RBR (bebas FIFO overrun)
            uint8_t *raw_img_buf = mb_cls_input->data.uint8;
            size_t total_received = 0;
            uint32_t idle_Loops = 0;
            const uint32_t max_idle_loops = 2500000; // ~1.2 detik pada bus APB 24MHz
            bool rx_timeout = false;

            while (total_received < PP2_IMAGE_BYTES) {
                if ((uart_regs[5] & 0x01u) != 0u) {
                    raw_img_buf[total_received++] = (uint8_t)(uart_regs[0] & 0xFFu);
                    idle_Loops = 0;
                } else {
                    idle_Loops++;
                    if (idle_Loops > max_idle_loops) {
                        rx_timeout = true;
                        break;
                    }
                }
            }

            if (rx_timeout) {
                xprintf("INFO - [ERROR] Timeout menerima gambar (%d/%d byte)!\r\n",
                        (int)total_received, PP2_IMAGE_BYTES);
                xprintf("PP2_ERR:TIMEOUT\r\n");
                continue;
            }

            // Terima 2 byte checksum (little-endian) langsung dari register hardware RBR
            uint8_t crc_bytes[2] = {0, 0};
            size_t crc_got = 0;
            idle_Loops = 0;
            while (crc_got < 2) {
                if ((uart_regs[5] & 0x01u) != 0u) {
                    crc_bytes[crc_got++] = (uint8_t)(uart_regs[0] & 0xFFu);
                    idle_Loops = 0;
                } else {
                    idle_Loops++;
                    if (idle_Loops > max_idle_loops) {
                        rx_timeout = true;
                        break;
                    }
                }
            }

            if (rx_timeout) {
                xprintf("INFO - [ERROR] Timeout menerima checksum!\r\n");
                xprintf("PP2_ERR:CRC_TIMEOUT\r\n");
                continue;
            }

            uint16_t expected_crc = (uint16_t)crc_bytes[0] | ((uint16_t)crc_bytes[1] << 8);

            // Hitung checksum, standar deviasi piksel (0..255), dan kuantisasi uint8 -> int8 (-128..127)
            uint32_t sum_bytes = 0;
            int64_t sum_sq = 0;
            int8_t *quant_buf = mb_cls_input->data.int8;

            for (size_t i = 0; i < PP2_IMAGE_BYTES; ++i) {
                uint32_t px = (uint32_t)raw_img_buf[i];
                sum_bytes += px;
                sum_sq += (int64_t)(px * px);
                quant_buf[i] = (int8_t)((int32_t)px - 128);
            }

            uint16_t calc_crc = (uint16_t)(sum_bytes & 0xFFFF);
            if (calc_crc != expected_crc) {
                xprintf("INFO - [ERROR] Checksum mismatch! Calc=0x%04X vs Rx=0x%04X\r\n",
                        (unsigned int)calc_crc, (unsigned int)expected_crc);
                xprintf("PP2_ERR:CHECKSUM\r\n");
                continue;
            }

            frame_counter++;

            float mean = (float)sum_bytes / (float)PP2_IMAGE_BYTES;
            float mean_sq = (float)sum_sq / (float)PP2_IMAGE_BYTES;
            float var = mean_sq - (mean * mean);
            float pixel_std = (var > 0.0f) ? sqrtf(var) : 0.0f;

            // Pastikan D-Cache untuk input tensor di-clean ke SRAM sebelum dibaca DMA Ethos-U55 NPU
            SCB_CleanDCache_by_Addr((uint32_t *)quant_buf, PP2_IMAGE_BYTES);

            // Jalankan inferensi di Arm Ethos-U55 NPU & ukur siklus PMU
            ETHOSU_PMU_CYCCNT_Reset(&ethosu_drv);
            ETHOSU_PMU_EVCNTR_ALL_Reset(&ethosu_drv);

            TfLiteStatus invoke_status = mb_cls_int_ptr->Invoke();
            if (invoke_status != kTfLiteOk) {
                xprintf("INFO - [ERROR] NPU Invoke() gagal pada frame #%d!\r\n", (int)frame_counter);
                xprintf("PP2_ERR:INVOKE\r\n");
                continue;
            }

            uint32_t npu_total_cycles  = ETHOSU_PMU_Get_CCNTR(&ethosu_drv);
            uint32_t npu_active_cycles = ETHOSU_PMU_Get_EVCNTR(&ethosu_drv, 0);
            uint32_t npu_axi0_rd       = ETHOSU_PMU_Get_EVCNTR(&ethosu_drv, 1);
            uint32_t npu_axi0_wr       = ETHOSU_PMU_Get_EVCNTR(&ethosu_drv, 2);
            uint32_t npu_idle_cycles   = ETHOSU_PMU_Get_EVCNTR(&ethosu_drv, 3);

            float npu_ms  = (float)npu_total_cycles / 400000.0f;
            float npu_fps = (npu_ms > 0.0f) ? (1000.0f / npu_ms) : 0.0f;

            // Dekuantisasi output [1, 2] INT8
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

            // Logika Keputusan Klasifikasi AI (Grove hanya menentukan class_id & confidence)
            uint8_t class_id = 2;
            float reported_conf_pct = 0.0f;
            const char *label_str = "undefined";
            const char *reason_str = "OK";

            if (pixel_std < BLANK_STD_THRESHOLD) {
                class_id = 2;
                reported_conf_pct = 0.0f; // 0.00% menandakan frame kosong/polos (NO EGG)
                label_str = "undefined";
                reason_str = "BLANK_FRAME";
            } else if (max_conf_pct < CONFIDENCE_THRESHOLD) {
                class_id = 2;
                reported_conf_pct = max_conf_pct; // > 0.00% menandakan ragu-ragu (< 80%)
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

            // Pecah confidence menjadi conf_int (0..100) dan conf_dec (0..99) untuk paket I2C 5-Byte
            int32_t conf_scaled = (int32_t)(reported_conf_pct * 100.0f + 0.5f);
            if (conf_scaled < 0) conf_scaled = 0;
            if (conf_scaled > 10000) conf_scaled = 10000;
            uint8_t conf_int = (uint8_t)(conf_scaled / 100);
            uint8_t conf_dec = (uint8_t)(conf_scaled % 100);

            // Susun Paket I2C 5-Byte Murni AI: [0xAA, class_id, conf_int, conf_dec, checksum]
            uint8_t i2c_pkt[5];
            i2c_pkt[0] = 0xAA;
            i2c_pkt[1] = class_id;
            i2c_pkt[2] = conf_int;
            i2c_pkt[3] = conf_dec;
            i2c_pkt[4] = (uint8_t)(i2c_pkt[0] ^ i2c_pkt[1] ^ i2c_pkt[2] ^ i2c_pkt[3]);

            // Kirim Paket 5-Byte ke ESP32 Slave (0x55) dengan proteksi timeout (~5 ms)
            const char *i2c_status_str = send_i2c_packet_safe(ESP32_I2C_SLAVE_ADDR, i2c_pkt, 5);

            // Format string desimal untuk laporan Serial & JSON
            char std_s[16], pb_s[16], pj_s[16], conf_s[16], ms_s[16], fps_s[16];
            format_float(std_s, sizeof(std_s), pixel_std, 2);
            format_float(pb_s, sizeof(pb_s), p_betina_pct, 2);
            format_float(pj_s, sizeof(pj_s), p_jantan_pct, 2);
            format_float(conf_s, sizeof(conf_s), reported_conf_pct, 2);
            format_float(ms_s, sizeof(ms_s), npu_ms, 2);
            format_float(fps_s, sizeof(fps_s), npu_fps, 1);

            xprintf("INFO - -----------------------------------------------------------------------------------------\r\n");
            xprintf("INFO - [FRAME #%03d] Gambar Diterima: %d B | Checksum: 0x%04X [OK] | Pixel Std: %s\r\n",
                    (int)frame_counter, PP2_IMAGE_BYTES, (unsigned int)calc_crc, std_s);
            xprintf("INFO -   Output Mentah NPU (INT8) : Class 0 (Betina) = %4d | Class 1 (Jantan) = %4d\r\n",
                    (int)raw_c0, (int)raw_c1);
            xprintf("INFO -   Probabilitas Terukur     : P(Betina) = %s%%     | P(Jantan) = %s%%\r\n",
                    pb_s, pj_s);
            xprintf("INFO -   Hasil Klasifikasi AI     : CLASS %d -> %s (Confidence: %s%% | Reason: %s)\r\n",
                    (int)class_id, label_str, conf_s, reason_str);
            xprintf("INFO -   Transmisi I2C ke ESP32   : [0x%02X, %d, %d, %d, 0x%02X] -> Slave 0x%02X (%s)\r\n",
                    (unsigned int)i2c_pkt[0], (int)i2c_pkt[1], (int)i2c_pkt[2], (int)i2c_pkt[3],
                    (unsigned int)i2c_pkt[4], ESP32_I2C_SLAVE_ADDR, i2c_status_str);
            xprintf("INFO -   Performa Hardware NPU    : %s ms (%s FPS) | Active: %u cyc | Total: %u cyc\r\n",
                    ms_s, fps_s, (unsigned int)npu_active_cycles, (unsigned int)npu_total_cycles);
            xprintf("INFO - -----------------------------------------------------------------------------------------\r\n");

            // Baris JSON terstruktur untuk di-parse oleh Notebook 3
            xprintf("PP2_JSON:{\"frame\":%u,\"class_id\":%d,\"label\":\"%s\",\"conf\":%s,\"p_betina\":%s,\"p_jantan\":%s,"
                    "\"pixel_std\":%s,\"reason\":\"%s\",\"raw_c0\":%d,\"raw_c1\":%d,"
                    "\"i2c_pkt\":[%u,%u,%u,%u,%u],\"i2c_status\":\"%s\","
                    "\"npu_ms\":%s,\"npu_fps\":%s,\"npu_active_cyc\":%u,\"npu_idle_cyc\":%u,\"npu_total_cyc\":%u,"
                    "\"axi0_rd\":%u,\"axi0_wr\":%u}\r\n",
                    (unsigned int)frame_counter,
                    (int)class_id,
                    label_str,
                    conf_s,
                    pb_s,
                    pj_s,
                    std_s,
                    reason_str,
                    (int)raw_c0,
                    (int)raw_c1,
                    (unsigned int)i2c_pkt[0],
                    (unsigned int)i2c_pkt[1],
                    (unsigned int)i2c_pkt[2],
                    (unsigned int)i2c_pkt[3],
                    (unsigned int)i2c_pkt[4],
                    i2c_status_str,
                    ms_s,
                    fps_s,
                    (unsigned int)npu_active_cycles,
                    (unsigned int)npu_idle_cycles,
                    (unsigned int)npu_total_cycles,
                    (unsigned int)npu_axi0_rd,
                    (unsigned int)npu_axi0_wr);

            xprintf("INFO - [READY] Menunggu upload gambar berikutnya...\r\n");
        }
    }

    return 0;
}

int cv_mb_cls_deinit()
{
    return 0;
}
