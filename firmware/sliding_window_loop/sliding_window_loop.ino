// tinyecg - continuous sliding-window anomaly loop on ESP32-S3 (replay of an MIT-BIH stream)
//
// Same logic as src/stream_sim.py:
//   new sample -> ring buffer; every SW_STRIDE samples with a full SW_WINDOW window:
//   z-score -> INT8 encode+decode (TFLM) -> reconstruction MSE -> flag if > threshold
//   -> alarm if >= K of the last M windows are flagged -> repeat.
//
// Sketch folder must contain: sliding_window_loop.ino, stream_data.h, ecg_ae_model.h
// (stream_data.h and ecg_ae_model.h are written by `python -m src.export_stream`).
//
// FAST_FORWARD 1: replay as fast as possible (quick check, ~1 min for 300 s of signal)
// FAST_FORWARD 0: real time - a 180 Hz timer feeds samples while the loop runs inference
//
// Open Serial Monitor (115200) and send any character to start / run again.

#define FAST_FORWARD 1
#define PRINT_ALL    1     // 1: print every window, 0: only flagged windows and alarms

#include <Arduino.h>
#include <math.h>
#include "esp_timer.h"
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "ecg_ae_model.h"
#include "stream_data.h"

constexpr int kArenaSize = 96 * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];

static tflite::MicroInterpreter* interp = nullptr;
static TfLiteTensor* in_t = nullptr;
static TfLiteTensor* out_t = nullptr;
static bool init_ok = false;

// ---- sample ring buffer (single producer, single consumer) ----
constexpr uint32_t RING = 1024;                   // power of two, > window + backlog during inference
static float ring[RING];
static volatile uint32_t head = 0;                // total samples written
static volatile uint32_t src_i = 0;               // next stream sample to emit
static esp_timer_handle_t tick_timer = nullptr;

static inline void pushSample(float v) {
  ring[head & (RING - 1)] = v;
  __sync_synchronize();
  head = head + 1;
}

static void onTick(void*) {                       // stands in for the ADC interrupt at 180 Hz
  uint32_t i = src_i;
  if (i < SW_LEN) { pushSample(SW_STREAM[i]); src_i = i + 1; }
}

// ---- run state ----
static bool running = false;
static uint32_t next_end = SW_WINDOW;             // window end (in samples) of the next evaluation
static uint32_t win_idx = 0;
static uint8_t recent[SW_M];
static bool prev_alarm = false;
static uint32_t n_flag = 0, n_alarm = 0, n_episodes = 0, n_truth = 0;
static uint32_t flag_mismatch = 0, alarm_mismatch = 0, overruns = 0;
static double rel_sum = 0; static float rel_max = 0;
static uint32_t inf_sum_us = 0, inf_max_us = 0, win_sum_us = 0, win_max_us = 0;

static void initModel() {
  const tflite::Model* model = tflite::GetModel(g_ecg_ae_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) { Serial.println("schema mismatch"); return; }
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter si(model, resolver, tensor_arena, kArenaSize);
  interp = &si;
  if (interp->AllocateTensors() != kTfLiteOk) { Serial.println("AllocateTensors failed"); return; }
  in_t = interp->input(0);
  out_t = interp->output(0);
  if (in_t->bytes != SW_WINDOW || out_t->bytes != SW_WINDOW) { Serial.println("tensor size != window"); return; }
  init_ok = true;
}

static void startRun() {
  head = 0; src_i = 0; next_end = SW_WINDOW; win_idx = 0;
  memset(recent, 0, sizeof(recent)); prev_alarm = false;
  n_flag = n_alarm = n_episodes = n_truth = 0;
  flag_mismatch = alarm_mismatch = overruns = 0;
  rel_sum = 0; rel_max = 0;
  inf_sum_us = inf_max_us = win_sum_us = win_max_us = 0;
  Serial.printf("\n== replay: record %d | preset %s (%d-of-%d) | threshold %.4f | %d windows | mode %s ==\n",
                SW_RECORD, SW_PRESET, SW_K, SW_M, (double)SW_THRESHOLD, SW_NUM_WIN,
                FAST_FORWARD ? "fast-forward" : "real-time");
  Serial.printf("arena used %u of %d bytes | in scale %.6f zp %d | out scale %.6f zp %d\n",
                (unsigned)interp->arena_used_bytes(), kArenaSize,
                in_t->params.scale, in_t->params.zero_point, out_t->params.scale, out_t->params.zero_point);
#if !FAST_FORWARD
  esp_timer_create_args_t ta = {};
  ta.callback = &onTick;
  ta.name = "adc180";
  esp_timer_create(&ta, &tick_timer);
  esp_timer_start_periodic(tick_timer, 1000000 / SW_FS);
#endif
  running = true;
}

static void finishRun() {
#if !FAST_FORWARD
  if (tick_timer) { esp_timer_stop(tick_timer); esp_timer_delete(tick_timer); tick_timer = nullptr; }
#endif
  running = false;
  uint32_t n = win_idx ? win_idx : 1;
  Serial.printf("\n%u windows | truth-anomalous %u | flagged %u | alarm windows %u | alarm episodes %u\n",
                (unsigned)win_idx, (unsigned)n_truth, (unsigned)n_flag, (unsigned)n_alarm, (unsigned)n_episodes);
  Serial.printf("desktop reference: flagged %d | alarm windows %d | alarm episodes %d\n",
                SW_REF_FLAGGED, SW_REF_ALARM_WINDOWS, SW_REF_EPISODES);
  Serial.printf("vs reference: flag mismatches %u | alarm-window mismatches %u | mean |err-ref|/ref %.1f%% (max %.1f%%)\n",
                (unsigned)flag_mismatch, (unsigned)alarm_mismatch, 100.0 * rel_sum / n, 100.0 * rel_max);
  Serial.printf("inference avg %u us, max %u us | full window step avg %u us, max %u us (budget %u us)\n",
                (unsigned)(inf_sum_us / n), (unsigned)inf_max_us,
                (unsigned)(win_sum_us / n), (unsigned)win_max_us, (unsigned)(1000000UL * SW_STRIDE / SW_FS));
  if (overruns) Serial.printf("WARNING: ring buffer overruns: %u\n", (unsigned)overruns);
  Serial.println("Send any character to run again.");
}

static void processWindow(uint32_t end) {
  static float w[SW_WINDOW];
  uint32_t t_start = micros();
  if (head - (end - SW_WINDOW) > RING) overruns++;
  double s = 0;
  for (int i = 0; i < SW_WINDOW; i++) { w[i] = ring[(end - SW_WINDOW + i) & (RING - 1)]; s += w[i]; }
  double mean = s / SW_WINDOW, v = 0;
  for (int i = 0; i < SW_WINDOW; i++) { double d = w[i] - mean; v += d * d; }
  double den = sqrt(v / SW_WINDOW) + 1e-6;

  for (int i = 0; i < SW_WINDOW; i++) {
    w[i] = (float)((w[i] - mean) / den);                                  // z-scored window
    int q = (int)roundf(w[i] / in_t->params.scale) + in_t->params.zero_point;
    in_t->data.int8[i] = (int8_t)constrain(q, -128, 127);
  }
  uint32_t t0 = micros();
  if (interp->Invoke() != kTfLiteOk) { Serial.println("Invoke failed"); return; }
  uint32_t inf_us = micros() - t0;

  double se = 0;
  for (int i = 0; i < SW_WINDOW; i++) {
    float y = (out_t->data.int8[i] - out_t->params.zero_point) * out_t->params.scale;
    double d = w[i] - y;
    se += d * d;
  }
  float err = (float)(se / SW_WINDOW);
  bool flag = err > SW_THRESHOLD;
  for (int i = 0; i < SW_M - 1; i++) recent[i] = recent[i + 1];            // k-of-m over the last M windows
  recent[SW_M - 1] = flag;
  int cnt = 0; for (int i = 0; i < SW_M; i++) cnt += recent[i];
  bool alarm = cnt >= SW_K;
  uint32_t step_us = micros() - t_start;

  n_flag += flag; n_alarm += alarm; n_episodes += (alarm && !prev_alarm); prev_alarm = alarm;
  bool truth = false;
  if (win_idx < SW_NUM_WIN) {
    truth = SW_TRUTH[win_idx]; n_truth += truth;
    float ref = SW_REF_ERR[win_idx];
    float rel = fabsf(err - ref) / (ref > 1e-9f ? ref : 1e-9f);
    rel_sum += rel; if (rel > rel_max) rel_max = rel;
    flag_mismatch += (flag != (bool)SW_REF_FLAG[win_idx]);
    alarm_mismatch += (alarm != (bool)SW_REF_ALARM[win_idx]);
  }
  inf_sum_us += inf_us; if (inf_us > inf_max_us) inf_max_us = inf_us;
  win_sum_us += step_us; if (step_us > win_max_us) win_max_us = step_us;

  if (PRINT_ALL || flag || alarm) {
    float t = SW_START_S + (float)end / SW_FS;
    Serial.printf("t=%7.1fs err=%.4f %s %s truth=%s\n", t, err, flag ? "flag " : "     ",
                  alarm ? "*** ALARM ***" : "             ", truth ? "ANOM" : "norm");
  }
  win_idx++;
}

void setup() {
  Serial.begin(115200);
  initModel();
}

void loop() {
  if (!running) {
    if (Serial.available()) {
      while (Serial.available()) Serial.read();
      if (init_ok) startRun();
    } else {
      static uint32_t last = 0;
      if (millis() - last > 3000) {
        last = millis();
        Serial.println(init_ok ? "tinyecg ready - send any character to start" : "model init failed (see earlier message)");
      }
      delay(20);
    }
    return;
  }

  if (head < next_end) {
#if FAST_FORWARD
    while (head < next_end && src_i < SW_LEN) pushSample(SW_STREAM[src_i++]);
#else
    delay(2);
#endif
    if (head < next_end && src_i >= SW_LEN) { finishRun(); return; }      // stream exhausted
    if (head < next_end) return;
  }
  processWindow(next_end);
  next_end += SW_STRIDE;
}
