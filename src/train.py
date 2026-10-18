"""Train the autoencoder on NORMAL windows only, then pick a threshold on validation-normal data."""
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import config as C
from .dataset import build_split
from .model import build_autoencoder, recon_error


def main():
    tf.keras.utils.set_random_seed(C.SEED)
    C.MODELS_DIR.mkdir(exist_ok=True); C.RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading train split..."); Xtr, ytr, _ = build_split("train")
    print("Loading val split...");   Xva, yva, _ = build_split("val")
    Xtr_n, Xva_n = Xtr[ytr == 0], Xva[yva == 0]
    print(f"train normal windows: {len(Xtr_n)} | val normal windows: {len(Xva_n)}")

    model = build_autoencoder()
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    cb = [keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)]
    hist = model.fit(Xtr_n, Xtr_n, validation_data=(Xva_n, Xva_n),
                     epochs=C.EPOCHS, batch_size=C.BATCH_SIZE, callbacks=cb, verbose=2)
    model.save(C.MODELS_DIR / "autoencoder.keras")

    thr = float(np.percentile(recon_error(model, Xva_n), C.THRESHOLD_PERCENTILE))
    with open(C.MODELS_DIR / "threshold_float.json", "w") as f:
        json.dump({"threshold": thr, "percentile": C.THRESHOLD_PERCENTILE}, f, indent=2)
    with open(C.RESULTS_DIR / "train_history.json", "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()}, f, indent=2)
    print(f"Saved model. Float threshold (p{C.THRESHOLD_PERCENTILE}) = {thr:.5f}")


if __name__ == "__main__":
    main()
