#include "FreeRTOS.h"
#include "task.h"
#include "pico/stdlib.h"
#include <stdio.h>
#include <string.h>

#include <hardware/gpio.h>
#include <hardware/i2c.h>

#include "mpu6050.h"

#include "edge-impulse-sdk/classifier/ei_model_types.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"
#include "model-parameters/model_metadata.h"

using namespace ei;

extern "C" EI_IMPULSE_ERROR
run_classifier(ei::signal_t *signal, ei_impulse_result_t *result, bool debug);

const int MPU_ADDRESS = 0x68;
const int I2C_SDA_GPIO = 4;
const int I2C_SCL_GPIO = 5;

const uint LED_R_GPIO = 14;
const uint LED_G_GPIO = 13;
const uint LED_B_GPIO = 15;

const float CONFIDENCE_THRESHOLD = 0.7f;

static void mpu6050_init(){
    i2c_init(i2c_default, 400 * 1000);
    gpio_set_function(I2C_SDA_GPIO, GPIO_FUNC_I2C);
    gpio_set_function(I2C_SCL_GPIO, GPIO_FUNC_I2C);
    gpio_pull_up(I2C_SDA_GPIO);
    gpio_pull_up(I2C_SCL_GPIO);

    uint8_t buf[] = { 0x6B, 0x00 };
    i2c_write_blocking(i2c_default, MPU_ADDRESS, buf, 2, false);
}

static void mpu6050_read_accel(int16_t accel[3]){
    uint8_t buffer[6];
    uint8_t reg = 0x3B;

    i2c_write_blocking(i2c_default, MPU_ADDRESS, &reg, 1, true);
    i2c_read_blocking(i2c_default, MPU_ADDRESS, buffer, 6, false);

    for (int i = 0; i < 3; i++) {
        accel[i] = (buffer[i * 2] << 8 | buffer[(i * 2) + 1]);
    }
}

static void led_rgb_init(){
    gpio_init(LED_R_GPIO);
    gpio_init(LED_G_GPIO);
    gpio_init(LED_B_GPIO);
    gpio_set_dir(LED_R_GPIO, GPIO_OUT);
    gpio_set_dir(LED_G_GPIO, GPIO_OUT);
    gpio_set_dir(LED_B_GPIO, GPIO_OUT);
    gpio_put(LED_R_GPIO, 0);
    gpio_put(LED_G_GPIO, 0);
    gpio_put(LED_B_GPIO, 0);
}

static void led_set(bool r, bool g, bool b){
    gpio_put(LED_R_GPIO, r);
    gpio_put(LED_G_GPIO, g);
    gpio_put(LED_B_GPIO, b);
}

static void gesture_recognize_task(void *p){
    mpu6050_init();
    led_rgb_init();

    int16_t accelerometer[3] = { 0 };

    while (true) {
        float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

        for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3) {
            mpu6050_read_accel(accelerometer);
            buffer[ix + 0] = accelerometer[0];
            buffer[ix + 1] = accelerometer[1];
            buffer[ix + 2] = accelerometer[2];

            vTaskDelay(pdMS_TO_TICKS(10));
        }

        ei::signal_t signal;
        numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);

        ei_impulse_result_t result = { 0 };
        run_classifier(&signal, &result, false);

        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("    %s: %.5f\n",
                result.classification[ix].label,
                result.classification[ix].value);
        }

        size_t best_ix = 0;
        float best_value = result.classification[0].value;
        for (size_t ix = 1; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            if (result.classification[ix].value > best_value) {
                best_value = result.classification[ix].value;
                best_ix = ix;
            }
        }

        const char *label = result.classification[best_ix].label;
        ei_printf(">> Classe vencedora: %s (%.3f)\n", label, best_value);

        if (best_value < CONFIDENCE_THRESHOLD) {
            led_set(0, 0, 0);
        } else if (strcmp(label, "Idle") == 0) {
            led_set(1, 0, 0);
        } else if (strcmp(label, "Updown") == 0) {
            led_set(0, 1, 0);
        } else if (strcmp(label, "Waving") == 0) {
            led_set(0, 0, 1);
        } else {
            led_set(0, 0, 0);
        }
    }
}

int main(void){
    stdio_init_all();
    xTaskCreate(gesture_recognize_task, "gesture_task", 8192, NULL, 1, NULL);
    vTaskStartScheduler();

    while (true);
}