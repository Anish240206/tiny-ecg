"""Plot original vs reconstruction for a few normal and anomalous TEST windows.
Run from repo root:  python -m scripts.plot_recon"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from src import config as C
from src.dataset import build_split

model = keras.models.load_model(C.MODELS_DIR / "autoencoder.keras")
X, y, rec = build_split("test")
rng = np.random.default_rng(0)
idx = list(rng.choice(np.where(y == 0)[0], 3, replace=False)) + \
      list(rng.choice(np.where(y == 1)[0], 3, replace=False))
rec_x = model.predict(X[idx], verbose=0)
t = np.arange(C.WINDOW) / C.FS
fig, ax = plt.subplots(2, 3, figsize=(14, 6), sharey=True)
for a, i, r in zip(ax.ravel(), idx, rec_x):
    a.plot(t, X[i, :, 0], label="input"); a.plot(t, r[:, 0], label="recon")
    mse = np.mean((X[i] - r) ** 2)
    a.set_title(f"rec {rec[i]} | {'ANOM' if y[i] else 'normal'} | mse={mse:.3f}")
ax[0, 0].legend(); plt.tight_layout()
C.RESULTS_DIR.mkdir(exist_ok=True)
plt.savefig(C.RESULTS_DIR / "recon_examples.png", dpi=150); plt.show()
