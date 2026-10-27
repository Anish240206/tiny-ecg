"""Fully-convolutional 1D autoencoder (translation-equivariant, so a QRS complex can sit
anywhere in the window). Bottleneck = 45 time steps x BOTTLENECK_CH channels.
Only Conv / TransposeConv ops -> supported by TFLM on the ESP32-S3."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from . import config as C
from .scoring import scores_from_sq


def build_autoencoder(window=C.WINDOW, bottleneck_ch=C.BOTTLENECK_CH):
    inp = keras.Input(shape=(window, 1), name="ecg")
    x = layers.Conv1D(16, 7, strides=2, padding="same", activation="relu")(inp)   # 180
    x = layers.Conv1D(32, 5, strides=2, padding="same", activation="relu")(x)     # 90
    z = layers.Conv1D(bottleneck_ch, 5, strides=2, padding="same", name="latent")(x)  # 45
    x = layers.Conv1DTranspose(32, 5, strides=2, padding="same", activation="relu")(z)  # 90
    x = layers.Conv1DTranspose(16, 5, strides=2, padding="same", activation="relu")(x)  # 180
    out = layers.Conv1DTranspose(1, 7, strides=2, padding="same", name="recon")(x)       # 360
    return keras.Model(inp, out, name="ecg_autoencoder")


def recon_error(model, X, batch_size=512, mode=None):
    """Per-window anomaly score from squared reconstruction error (see scoring.py)."""
    pred = model.predict(X, batch_size=batch_size, verbose=0)
    return scores_from_sq(((X - pred) ** 2)[..., 0], mode)


if __name__ == "__main__":
    m = build_autoencoder()
    m.summary()
    print("approx INT8 size (KB):", m.count_params() / 1024)
