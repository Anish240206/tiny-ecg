"""Evaluate the float model on held-out patients (DS2).
Thresholds are chosen on validation-normal windows only (never on test)."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow import keras

from . import config as C
from .dataset import build_split
from .metrics import summarize, plot_diagnostics, sweep_thresholds
from .scoring import scores_from_sq


def sq_error(model, X):
    pred = model.predict(X, batch_size=512, verbose=0)
    return ((X - pred) ** 2)[..., 0]


def main():
    model = keras.models.load_model(C.MODELS_DIR / "autoencoder.keras")
    print("Loading splits...")
    Xv, yv, _ = build_split("val")
    X, y, rec = build_split("test")
    sq = sq_error(model, X)

    print(f"\nAUROC by score type (threshold-free), configured mode = {C.SCORE_MODE}:")
    for m in ("mean", "peak"):
        print(f"  {m:5s} {roc_auc_score(y, scores_from_sq(sq, m)):.4f}")

    err = scores_from_sq(sq)
    err_val_n = scores_from_sq(sq_error(model, Xv[yv == 0]))
    thr = float(np.percentile(err_val_n, C.THRESHOLD_PERCENTILE))
    res = summarize(y, err, thr)
    plot_diagnostics(y, err, thr, "float")
    json.dump(res, open(C.RESULTS_DIR / "float_metrics.json", "w"), indent=2)
    json.dump({"threshold": thr, "percentile": C.THRESHOLD_PERCENTILE, "mode": C.SCORE_MODE},
              open(C.MODELS_DIR / "threshold_float.json", "w"), indent=2)
    print(f"\nMain result (threshold = val p{C.THRESHOLD_PERCENTILE}):")
    for k, v in res.items():
        print(f"  {k:24s} {v}")

    rows = sweep_thresholds(err_val_n, y, err)
    json.dump(rows, open(C.RESULTS_DIR / "operating_points_float.json", "w"), indent=2)
    print("\nOperating points (threshold from val percentile):")
    print(f"  {'val_pct':>7s} {'recall':>8s} {'specif':>8s} {'precis':>8s} {'FA/hour':>9s}")
    for r in rows:
        print(f"  {r['val_percentile']:7d} {r['recall_sensitivity']:8.3f} {r['specificity']:8.3f} "
              f"{r['precision']:8.3f} {r['false_alarms_per_hour']:9.1f}")

    print("\nPer-record AUROC (records with both classes):")
    for r in np.unique(rec):
        m = rec == r
        if 0 < y[m].sum() < m.sum():
            print(f"  {r}: {roc_auc_score(y[m], err[m]):.3f}  (anomalous {int(y[m].sum())}/{int(m.sum())})")


if __name__ == "__main__":
    main()
