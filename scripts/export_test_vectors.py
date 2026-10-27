"""Export known input/output pairs for the INT8 model as a C header, so the ESP32-S3
firmware can be checked bit-exactly against Python.
Run from repo root:  python -m scripts.export_test_vectors"""
import numpy as np
from src import config as C
from src.dataset import build_split
from src.tflite_utils import TFLiteAE

N_EACH = 4
q = TFLiteAE(C.MODELS_DIR / "autoencoder_int8.tflite")
X, y, rec = build_split("test")
rng = np.random.default_rng(C.SEED)
idx = list(rng.choice(np.where(y == 0)[0], N_EACH, replace=False)) + \
      list(rng.choice(np.where(y == 1)[0], N_EACH, replace=False))

ins, outs, errs, labels = [], [], [], []
for i in idx:
    x = X[i]
    qi = np.clip(np.round(x / q.in_scale + q.in_zp), -128, 127).astype(np.int8)
    q.interp.set_tensor(q.inp["index"], qi[None]); q.interp.invoke()
    qo = q.interp.get_tensor(q.out["index"])[0].astype(np.int8)
    rec_f = (qo.astype(np.float32) - q.out_zp) * q.out_scale
    ins.append(qi[:, 0]); outs.append(qo[:, 0])
    errs.append(float(np.mean((x - rec_f) ** 2))); labels.append(int(y[i]))

def arr2d(name, a):
    rows = ",\n".join("  {" + ", ".join(str(int(v)) for v in r) + "}" for r in a)
    return f"const int8_t {name}[{len(a)}][{C.WINDOW}] = {{\n{rows}\n}};\n"

with open(C.MODELS_DIR / "test_vectors.h", "w") as f:
    f.write("#pragma once\n#include <stdint.h>\n\n")
    f.write(f"#define TV_COUNT {len(idx)}\n#define TV_WINDOW {C.WINDOW}\n")
    f.write(f"const float TV_IN_SCALE = {q.in_scale:.9g}f;\nconst int TV_IN_ZP = {int(q.in_zp)};\n")
    f.write(f"const float TV_OUT_SCALE = {q.out_scale:.9g}f;\nconst int TV_OUT_ZP = {int(q.out_zp)};\n")
    f.write("const int TV_LABEL[TV_COUNT] = {" + ", ".join(map(str, labels)) + "};\n")
    f.write("const float TV_EXPECTED_ERR[TV_COUNT] = {" + ", ".join(f"{e:.9g}f" for e in errs) + "};\n\n")
    f.write("// z-scored window, quantized with TV_IN_SCALE / TV_IN_ZP\n" + arr2d("TV_INPUT", ins) + "\n")
    f.write("// expected raw INT8 model output for each input\n" + arr2d("TV_OUTPUT", outs))
print("Wrote models/test_vectors.h | labels:", labels)
print("expected errors:", [round(e, 4) for e in errs])
