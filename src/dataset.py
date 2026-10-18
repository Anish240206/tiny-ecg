"""MIT-BIH loading, filtering, windowing, labelling."""
import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

from . import config as C


def load_record(rec: int):
    """Return (MLII signal @360 Hz, beat sample indices @360 Hz, beat symbols)."""
    path = str(C.DATA_DIR / str(rec))
    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    ch = record.sig_name.index("MLII")          # channel order differs between records
    sig = record.p_signal[:, ch].astype(np.float64)
    keep = [i for i, s in enumerate(ann.symbol) if s in C.BEAT_SYMBOLS]
    samples = np.asarray(ann.sample)[keep]
    symbols = np.asarray(ann.symbol)[keep]
    return sig, samples, symbols


def preprocess(sig, beat_samples):
    """Band-pass 0.5-40 Hz, decimate to 180 Hz. Beat indices are rescaled to match."""
    sos = butter(3, C.BAND, btype="band", fs=C.FS_RAW, output="sos")
    filt = sosfiltfilt(sos, sig)
    return filt[::C.DECIM].astype(np.float32), beat_samples // C.DECIM


def window_signal(sig, beat_idx, beat_sym, stride):
    """Cut sliding windows; label 0 = every beat is 'N', 1 = contains a non-N beat.
    Windows with no beats are dropped. Each window is z-scored on its own."""
    W = C.WINDOW
    if len(sig) < W:
        return np.empty((0, W, 1), np.float32), np.empty(0, np.int8), np.empty(0, np.int64)
    starts = np.arange(0, len(sig) - W + 1, stride)
    nonN = (beat_sym != C.NORMAL_SYMBOL).astype(np.int64)
    cs = np.concatenate([[0], np.cumsum(nonN)])
    lo = np.searchsorted(beat_idx, starts, side="left")
    hi = np.searchsorted(beat_idx, starts + W, side="left")
    count = hi - lo
    bad = cs[hi] - cs[lo]
    ok = count > 0
    starts, bad = starts[ok], bad[ok]
    X = np.stack([sig[s:s + W] for s in starts])
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)
    return X[..., None].astype(np.float32), (bad > 0).astype(np.int8), starts


def build_split(name: str, use_cache=True):
    """Returns X (n,W,1), y (n,), rec_ids (n,). Cached under data/processed/."""
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    cache = C.PROC_DIR / f"{name}.npz"
    if use_cache and cache.exists():
        d = np.load(cache)
        return d["X"], d["y"], d["rec"]
    records, stride = C.SPLITS[name]
    Xs, ys, rs = [], [], []
    for rec in records:
        sig, bs, sym = load_record(rec)
        sig, bidx = preprocess(sig, bs)
        X, y, _ = window_signal(sig, bidx, sym, stride)
        Xs.append(X); ys.append(y); rs.append(np.full(len(y), rec))
        print(f"  record {rec}: {len(y)} windows, {int(y.sum())} anomalous")
    X, y, rec = np.concatenate(Xs), np.concatenate(ys), np.concatenate(rs)
    np.savez_compressed(cache, X=X, y=y, rec=rec)
    return X, y, rec
