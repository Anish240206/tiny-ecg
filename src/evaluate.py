"""Evaluate the float model on held-out patients (DS2)."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow import keras

from . import config as C
from .dataset import build_split
from .metrics import summarize, plot_diagnostics
from .model import recon_error


def main():
    model = keras.models.load_model(C.MODELS_DIR / "autoencoder.keras")
    thr = json.load(open(C.MODELS_DIR / "threshold_float.json"))["threshold"]
    print("Loading test split..."); X, y, rec = build_split("test")
    err = recon_error(model, X)
    res = summarize(y, err, thr)
    plot_diagnostics(y, err, thr, "float")
    json.dump(res, open(C.RESULTS_DIR / "float_metrics.json", "w"), indent=2)
    for k, v in res.items():
        print(f"{k:24s} {v}")
    print("\nPer-record AUROC (records with both classes):")
    for r in np.unique(rec):
        m = rec == r
        if 0 < y[m].sum() < m.sum():
            print(f"  {r}: {roc_auc_score(y[m], err[m]):.3f}  (anomalous {int(y[m].sum())}/{int(m.sum())})")


if __name__ == "__main__":
    main()
