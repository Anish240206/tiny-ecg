"""Show what MIT-BIH annotates inside a time window of a record (rhythm labels, artifacts, beat types).

Place in src/ and run from the repo root (venv active):

    python -m src.rhythm_check --record 210 --start 215 --end 260

Why: the project's ground truth is per beat ("window is anomalous if it holds a non-N beat"). A sustained
abnormal RHYTHM (e.g. atrial fibrillation) whose beats are all labelled N, or a noisy stretch, is invisible
to that label even though the model reacts to it.
"""
import argparse
from collections import Counter

import wfdb

from . import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=int, default=210)
    ap.add_argument("--start", type=float, default=215, help="seconds")
    ap.add_argument("--end", type=float, default=260, help="seconds")
    args = ap.parse_args()

    ann = wfdb.rdann(str(C.DATA_DIR / str(args.record)), "atr")
    t = [s / C.FS_RAW for s in ann.sample]
    notes = ann.aux_note

    print(f"record {args.record}: rhythm / signal-quality annotations up to {args.end:.0f} s")
    for ti, sym, note in zip(t, ann.symbol, notes):
        if sym in ("+", "~", "|", "!", "[", "]", '"') and ti <= args.end:
            print(f"  t={ti:8.1f}s  symbol {sym!r:5} {note.strip()}")

    inside = [(ti, sym) for ti, sym in zip(t, ann.symbol) if args.start <= ti <= args.end]
    print(f"\nannotations between {args.start:.0f} s and {args.end:.0f} s: {len(inside)} in total")
    for sym, n in Counter(s for _, s in inside).most_common():
        kind = "beat" if sym in C.BEAT_SYMBOLS else "non-beat"
        print(f"  {sym!r:5} x{n:3}  ({kind}{', counted normal' if sym == C.NORMAL_SYMBOL else ''})")


if __name__ == "__main__":
    main()
