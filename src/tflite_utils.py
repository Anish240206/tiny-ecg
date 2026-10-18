"""Run a full-integer TFLite autoencoder from Python (mirrors what TFLM does on-device)."""
import numpy as np
import tensorflow as tf


class TFLiteAE:
    def __init__(self, path):
        self.interp = tf.lite.Interpreter(model_path=str(path))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]

    def reconstruct(self, x):
        """x: (W,1) float32 -> reconstruction (W,1) float32."""
        q = np.clip(np.round(x / self.in_scale + self.in_zp), -128, 127).astype(np.int8)
        self.interp.set_tensor(self.inp["index"], q[None])
        self.interp.invoke()
        o = self.interp.get_tensor(self.out["index"])[0].astype(np.float32)
        return (o - self.out_zp) * self.out_scale

    def error(self, x):
        return float(np.mean((x - self.reconstruct(x)) ** 2))

    def errors(self, X):
        return np.array([self.error(x) for x in X], dtype=np.float32)
