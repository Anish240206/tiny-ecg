"""Shared constants. Change things here, not in the other files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "mitdb"
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# --- Signal ---
FS_RAW = 360            # MIT-BIH sampling rate
DECIM = 2               # 360 -> 180 Hz (after 40 Hz low-pass this loses nothing useful)
FS = FS_RAW // DECIM    # 180 Hz
BAND = (0.5, 40.0)      # band-pass in Hz

# --- Windowing (same windows the ESP32 sliding-window loop will use) ---
WINDOW_S = 2.0
WINDOW = int(WINDOW_S * FS)        # 360 samples
STRIDE_TRAIN_S = 0.5               # overlap for more training data
STRIDE_EVAL_S = 1.0                # = hop used by the live loop
STRIDE_TRAIN = int(STRIDE_TRAIN_S * FS)
STRIDE_EVAL = int(STRIDE_EVAL_S * FS)

# --- Inter-patient split (de Chazal DS1/DS2). Paced records 102,104,107,217 excluded. ---
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
VAL_RECORDS = [215, 220, 223, 230]                 # held out of DS1 for threshold selection
TRAIN_RECORDS = [r for r in DS1 if r not in VAL_RECORDS]
TEST_RECORDS = DS2

SPLITS = {"train": (TRAIN_RECORDS, STRIDE_TRAIN),
          "val": (VAL_RECORDS, STRIDE_EVAL),
          "test": (TEST_RECORDS, STRIDE_EVAL)}

# Annotation symbols that are actual heartbeats. Only 'N' counts as normal.
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
NORMAL_SYMBOL = "N"

# --- Training / threshold ---
BOTTLENECK_CH = 2   # latent = 45 steps x BOTTLENECK_CH channels
EPOCHS = 100
BATCH_SIZE = 128
SEED = 42
THRESHOLD_PERCENTILE = 99.0   # of reconstruction error on validation-normal windows
