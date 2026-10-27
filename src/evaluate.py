"""Evaluate the float model on held-out patients (DS2)."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow import keras

from . import config as C
from .dataset import build_split
from .metrics import summarize, plot_diagnostics
from .model import recon_error
from .scoring import scores_from_sq


def main():
    model = keras.models.load_model(C.MODELS_DIR / "autoencoder.keras")
    thr = json.load(open(C.MODELS_DIR / "threshold_float.json"))["threshold"]
    print("Loading test split..."); X, y, rec = build_split("test")
    pred = model.predict(X, batch_size=512, verbose=0)
    sq = ((X - pred) ** 2)[..., 0]
    print(f"Score mode = {C.SCORE_MODE}.  AUROC by score type (threshold-free):")
    for m in ("mean", "peak"):
        print(f"  {m:5s} {roc_auc_score(y, scores_from_sq(sq, m)):.4f}")
    err = scores_from_sq(sq)
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
