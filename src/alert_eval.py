"""Deployment-level evaluation of the INT8 model: window flags -> k-of-m alert rule
-> scored on non-overlapping 10 s segments (a segment is anomalous if any window in it is).

Thresholds come from validation-normal scores only. Run:  python -m src.alert_eval"""
import json
import numpy as np

from . import config as C
from .dataset import build_split
from .tflite_utils import TFLiteAE

SEG = 10                                   # windows per segment (hop = 1 s -> 10 s)
PERCENTILES = (90, 95, 97, 99)
RULES = [(1, 1), (2, 3), (3, 5), (4, 5)]   # (k, m): alarm if >= k of last m windows flagged


def kofm(flags, k, m):
    """Rolling count over the last m windows (shorter at the start of a record)."""
    c = np.concatenate([[0], np.cumsum(flags)])
    idx = np.arange(1, len(flags) + 1)
    return (c[idx] - c[np.maximum(idx - m, 0)]) >= k


def segment_metrics(y, alarm, rec):
    tp = fn = fp = tn = 0
    for r in np.unique(rec):
        yy, aa = y[rec == r], alarm[rec == r]
        for s in range(0, len(yy) - SEG + 1, SEG):
            truth, hit = yy[s:s + SEG].any(), aa[s:s + SEG].any()
            tp += truth and hit; fn += truth and not hit
            fp += (not truth) and hit; tn += (not truth) and not hit
    hours_normal = (fp + tn) * SEG / 3600.0
    return {"segment_recall": tp / max(tp + fn, 1),
            "segment_specificity": tn / max(tn + fp, 1),
            "false_alarms_per_hour": fp / max(hours_normal, 1e-9),
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}


def main():
    q = TFLiteAE(C.MODELS_DIR / "autoencoder_int8.tflite")
    Xv, yv, _ = build_split("val")
    err_val_n = q.errors(Xv[yv == 0])
    X, y, rec = build_split("test")
    err = q.errors(X)

    rows = []
    print(f"{'val_pct':>7s} {'k/m':>5s} {'seg_recall':>11s} {'seg_spec':>9s} {'FA/hour':>9s}")
    for p in PERCENTILES:
        thr = float(np.percentile(err_val_n, p))
        flags = err > thr
        for k, m in RULES:
            alarm = np.zeros_like(flags)
            for r in np.unique(rec):
                sel = rec == r
                alarm[sel] = kofm(flags[sel], k, m)
            res = segment_metrics(y, alarm, rec)
            res.update(val_percentile=p, k=k, m=m, threshold=thr)
            rows.append(res)
            print(f"{p:7d} {k}/{m:<3d} {res['segment_recall']:11.3f} "
                  f"{res['segment_specificity']:9.3f} {res['false_alarms_per_hour']:9.1f}")
    json.dump(rows, open(C.RESULTS_DIR / "alert_operating_points_int8.json", "w"), indent=2)


if __name__ == "__main__":
    main()
