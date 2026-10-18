"""Small 1D-CNN autoencoder. Decoder is Dense-only on purpose: Dense/Conv/Reshape
are the safest ops for TFLM on the ESP32-S3."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from . import config as C


def build_autoencoder(window=C.WINDOW, latent=C.LATENT_DIM):
    inp = keras.Input(shape=(window, 1), name="ecg")
    x = layers.Conv1D(8, 7, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv1D(16, 5, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv1D(16, 5, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    z = layers.Dense(latent, activation="relu", name="latent")(x)
    x = layers.Dense(32, activation="relu")(z)
    x = layers.Dense(window)(x)
    out = layers.Reshape((window, 1), name="recon")(x)
    return keras.Model(inp, out, name="ecg_autoencoder")


def recon_error(model, X, batch_size=512):
    """Per-window mean squared reconstruction error."""
    pred = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean((X - pred) ** 2, axis=(1, 2))


if __name__ == "__main__":
    m = build_autoencoder()
    m.summary()
    print("approx INT8 size (KB):", m.count_params() / 1024)
