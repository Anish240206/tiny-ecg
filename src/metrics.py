import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C


def summarize(y, err, thr):
    """y: 1 = anomalous window. Flag anomaly when err > thr."""
    pred = (err > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    recall = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    normal_hours = (tn + fp) * C.STRIDE_EVAL_S / 3600.0   # windows hop by STRIDE_EVAL_S
    return {
        "threshold": float(thr),
        "auroc": float(roc_auc_score(y, err)),
        "recall_sensitivity": recall,
        "specificity": spec,
        "precision": precision,
        "false_alarms_per_hour": fp / max(normal_hours, 1e-9),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "n_windows": int(len(y)), "anomaly_fraction": float(y.mean()),
    }


def plot_diagnostics(y, err, thr, tag):
    C.RESULTS_DIR.mkdir(exist_ok=True)
    fpr, tpr, _ = roc_curve(y, err)
    plt.figure(figsize=(4.5, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"ROC ({tag})"); plt.tight_layout()
    plt.savefig(C.RESULTS_DIR / f"roc_{tag}.png", dpi=150); plt.close()

    hi = np.percentile(err, 99.5)
    bins = np.linspace(0, hi, 80)
    plt.figure(figsize=(6, 4))
    plt.hist(err[y == 0], bins, alpha=.6, label="normal", density=True)
    plt.hist(err[y == 1], bins, alpha=.6, label="anomalous", density=True)
    plt.axvline(thr, c="k", ls="--", label="threshold")
    plt.xlabel("Reconstruction MSE"); plt.legend(); plt.title(f"Error distribution ({tag})")
    plt.tight_layout(); plt.savefig(C.RESULTS_DIR / f"errhist_{tag}.png", dpi=150); plt.close()


def sweep_thresholds(err_val_normal, y, err, percentiles=(80, 90, 95, 97, 98, 99)):
    """Thresholds come from VALIDATION-normal scores only; metrics are measured on test."""
    rows = []
    for p in percentiles:
        r = summarize(y, err, float(np.percentile(err_val_normal, p)))
        r["val_percentile"] = p
        rows.append(r)
    return rows
