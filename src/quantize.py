"""Post-training full-integer INT8 quantization -> .tflite -> C header for TFLM.
Re-picks the threshold with the INT8 model, then compares float vs INT8 on DS2."""
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import config as C
from .dataset import build_split
from .metrics import summarize, plot_diagnostics
from .tflite_utils import TFLiteAE


def to_c_header(tflite_bytes, path, name="g_ecg_ae_model"):
    lines = []
    for i in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[i:i + 12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    with open(path, "w") as f:
        f.write("#pragma once\n#include <stdint.h>\n\n")
        f.write(f"alignas(16) const unsigned char {name}[] = {{\n")
        f.write("\n".join(lines) + "\n};\n")
        f.write(f"const unsigned int {name}_len = {len(tflite_bytes)};\n")


def main():
    model = keras.models.load_model(C.MODELS_DIR / "autoencoder.keras")
    Xtr, ytr, _ = build_split("train")
    Xtr_n = Xtr[ytr == 0]
    rng = np.random.default_rng(C.SEED)
    rep = Xtr_n[rng.choice(len(Xtr_n), size=min(500, len(Xtr_n)), replace=False)]

    def rep_data():
        for x in rep:
            yield [x[None].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()

    tfl_path = C.MODELS_DIR / "autoencoder_int8.tflite"
    tfl_path.write_bytes(tfl)
    to_c_header(tfl, C.MODELS_DIR / "ecg_ae_model.h")
    print(f"INT8 .tflite size: {len(tfl)/1024:.1f} KB (target < 50 KB)")

    q = TFLiteAE(tfl_path)

    # threshold from validation-normal windows, using the INT8 model itself
    Xva, yva, _ = build_split("val")
    err_val = q.errors(Xva[yva == 0])
    thr = float(np.percentile(err_val, C.THRESHOLD_PERCENTILE))
    json.dump({"threshold": thr, "percentile": C.THRESHOLD_PERCENTILE},
              open(C.MODELS_DIR / "threshold_int8.json", "w"), indent=2)

    X, y, _ = build_split("test")
    err = q.errors(X)
    res = summarize(y, err, thr)
    res["model_size_kb"] = len(tfl) / 1024
    plot_diagnostics(y, err, thr, "int8")
    json.dump(res, open(C.RESULTS_DIR / "int8_metrics.json", "w"), indent=2)

    try:
        f = json.load(open(C.RESULTS_DIR / "float_metrics.json"))
        print(f"\n{'metric':24s}{'float':>12s}{'int8':>12s}")
        for k in ["auroc", "recall_sensitivity", "specificity", "false_alarms_per_hour"]:
            print(f"{k:24s}{f[k]:12.4f}{res[k]:12.4f}")
    except FileNotFoundError:
        print(res)


if __name__ == "__main__":
    main()
