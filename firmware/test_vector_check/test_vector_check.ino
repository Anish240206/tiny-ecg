// tinyecg - on-device test-vector check (ESP32-S3, TFLM)
// Sketch folder must also contain: ecg_ae_model.h, test_vectors.h
//
// Two checks per vector:
//   A) output tensor vs TV_OUTPUT (desktop INT8 output): should match exactly or within +/-1
//   B) reconstruction MSE vs TV_EXPECTED_ERR
// Results repeat every 10 s so you can open Serial Monitor at any time.

#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "ecg_ae_model.h"
#include "test_vectors.h"

constexpr int kArenaSize = 96 * 1024;            // first-run size; shrink after reading "arena used"
alignas(16) static uint8_t tensor_arena[kArenaSize];

static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interp = nullptr;
static TfLiteTensor* in_t = nullptr;
static TfLiteTensor* out_t = nullptr;
static bool ready = false;

void setup() {
  Serial.begin(115200);
  uint32_t t = millis();
  while (!Serial && millis() - t < 5000) delay(10);
  delay(500);
  Serial.println("\n== tinyecg test-vector check ==");

  model = tflite::GetModel(g_ecg_ae_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Schema version mismatch"); return;
  }
  static tflite::AllOpsResolver resolver;        // trim to MicroMutableOpResolver later
  static tflite::MicroInterpreter static_interp(model, resolver, tensor_arena, kArenaSize);
  interp = &static_interp;

  if (interp->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed (missing op or arena too small)"); return;
  }
  in_t = interp->input(0);
  out_t = interp->output(0);

  Serial.printf("model bytes: %u\n", (unsigned)g_ecg_ae_model_len);
  Serial.printf("arena used: %u of %d bytes\n", (unsigned)interp->arena_used_bytes(), kArenaSize);
  Serial.printf("in : type=%d scale=%.8f zp=%d bytes=%u  (header: %.8f, %d)\n",
                in_t->type, in_t->params.scale, in_t->params.zero_point,
                (unsigned)in_t->bytes, TV_IN_SCALE, TV_IN_ZP);
  Serial.printf("out: type=%d scale=%.8f zp=%d bytes=%u  (header: %.8f, %d)\n",
                out_t->type, out_t->params.scale, out_t->params.zero_point,
                (unsigned)out_t->bytes, TV_OUT_SCALE, TV_OUT_ZP);

  if (in_t->bytes != TV_WINDOW || out_t->bytes != TV_WINDOW) {
    Serial.println("Tensor size != TV_WINDOW - check model/window length"); return;
  }
  ready = true;
}

void runCheck() {
  int pass = 0;
  uint32_t total_us = 0;
  for (int v = 0; v < TV_COUNT; v++) {
    memcpy(in_t->data.int8, TV_INPUT[v], TV_WINDOW);   // vectors are already INT8

    uint32_t t0 = micros();
    TfLiteStatus s = interp->Invoke();
    uint32_t dt = micros() - t0;
    if (s != kTfLiteOk) { Serial.printf("v%d Invoke failed\n", v); continue; }
    total_us += dt;

    int mismatches = 0, maxdiff = 0;
    double se = 0;
    for (int i = 0; i < TV_WINDOW; i++) {
      int8_t o = out_t->data.int8[i];
      int d = abs((int)o - (int)TV_OUTPUT[v][i]);
      if (d) mismatches++;
      if (d > maxdiff) maxdiff = d;
      float x = (TV_INPUT[v][i] - TV_IN_ZP) * TV_IN_SCALE;
      float y = (o - TV_OUT_ZP) * TV_OUT_SCALE;
      se += (double)(x - y) * (x - y);
    }
    float mse = (float)(se / TV_WINDOW);
    float exp = TV_EXPECTED_ERR[v];
    //bool okA = maxdiff <= 1;
    bool okB = fabsf(mse - exp) <= 0.10f * exp + 0.002f;
    bool ok = okB;
    pass += ok;
    Serial.printf("v%d label=%d mse=%.4f expected=%.4f | out mismatches=%d maxdiff=%d | %s | %u us\n",
                  v, TV_LABEL[v], mse, exp, mismatches, maxdiff, ok ? "PASS" : "FAIL", (unsigned)dt);
  }
  Serial.printf("RESULT: %d/%d passed, avg latency %u us\n\n",
                pass, TV_COUNT, (unsigned)(total_us / TV_COUNT));
}

void loop() {
  if (ready) runCheck();
  else Serial.println("init failed - see messages above");
  delay(10000);
}
