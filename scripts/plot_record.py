"""Sanity check: plot 10 s of a record with beat annotations.
Run from repo root:  python -m scripts.plot_record 100"""
import sys
import matplotlib.pyplot as plt
from src import config as C
from src.dataset import load_record

rec = int(sys.argv[1]) if len(sys.argv) > 1 else 100
sig, samples, symbols = load_record(rec)
n = 10 * C.FS_RAW
plt.figure(figsize=(12, 3.5)); plt.plot(sig[:n], lw=.8)
for s, c in zip(samples, symbols):
    if s < n:
        plt.text(s, sig[s], c, color="r", ha="center")
plt.title(f"MIT-BIH record {rec} (MLII), first 10 s"); plt.tight_layout()
C.RESULTS_DIR.mkdir(exist_ok=True)
plt.savefig(C.RESULTS_DIR / f"record_{rec}.png", dpi=150); plt.show()
