"""
Deepfake Audio Detection — Stage 1: Feature Extraction (FAST VERSION)
======================================================================
Uses joblib parallel processing — 4x-8x faster than sequential.
Also supports --max_files to limit dataset size for quick testing.

Usage:
  python deepfake_features.py                    # all files, parallel
  python deepfake_features.py --max_files 1000   # only 1000 files
  python deepfake_features.py --workers 4        # 4 parallel workers
"""

import sys
import time
import logging
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("deepfake_features.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("Deepfake-Features")
SEP = "=" * 60

MANIFEST_CSV = "data_manifest.csv"
SR           = 16000
DURATION     = 3.0
N_MFCC       = 40
N_MELS       = 64
HOP_LENGTH   = 512


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_files", type=int, default=None,
                   help="Max files per split (e.g. 1000 for quick test)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers (default: 4)")
    return p.parse_args()


def extract_one(fpath: str) -> np.ndarray:
    try:
        import librosa
        y, _ = librosa.load(fpath, sr=SR, duration=DURATION, mono=True)
        tlen = int(SR * DURATION)
        y    = np.pad(y, (0, max(0, tlen - len(y))), mode="constant")[:tlen]
        hop  = HOP_LENGTH
        f    = []

        mfcc    = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=hop)
        mfcc_d  = librosa.feature.delta(mfcc)
        mfcc_dd = librosa.feature.delta(mfcc, order=2)
        for i in range(N_MFCC):
            f.extend([mfcc[i].mean(), mfcc[i].std()])
            f.extend([mfcc_d[i].mean(), mfcc_d[i].std()])
            f.extend([mfcc_dd[i].mean(), mfcc_dd[i].std()])

        mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=hop)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        f.extend([mel_db.mean(), mel_db.std(), mel_db.max(), mel_db.min()])

        chroma = librosa.feature.chroma_stft(y=y, sr=SR, hop_length=hop)
        for i in range(12):
            f.extend([chroma[i].mean(), chroma[i].std()])

        f.extend([
            librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=hop).mean(),
            librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=hop).std(),
            librosa.feature.spectral_bandwidth(y=y, sr=SR, hop_length=hop).mean(),
            librosa.feature.spectral_bandwidth(y=y, sr=SR, hop_length=hop).std(),
            librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=hop).mean(),
            librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=hop).std(),
            librosa.feature.spectral_contrast(y=y, sr=SR, hop_length=hop).mean(),
            librosa.feature.spectral_contrast(y=y, sr=SR, hop_length=hop).std(),
            librosa.feature.spectral_flatness(y=y, hop_length=hop).mean(),
            librosa.feature.spectral_flatness(y=y, hop_length=hop).std(),
            librosa.feature.zero_crossing_rate(y, hop_length=hop).mean(),
            librosa.feature.zero_crossing_rate(y, hop_length=hop).std(),
            librosa.feature.rms(y=y, hop_length=hop).mean(),
            librosa.feature.rms(y=y, hop_length=hop).std(),
        ])

        try:
            h = librosa.effects.harmonic(y)
            t = librosa.feature.tonnetz(y=h, sr=SR)
            f.extend([t.mean(), t.std()])
        except Exception:
            f.extend([0.0, 0.0])

        return np.array(f, dtype=np.float32)
    except Exception:
        return None


def process_split(split_df, split_name, max_files, n_workers):
    if max_files:
        per_class = max_files // 2
        real_df   = split_df[split_df["label"] == "real"].head(per_class)
        fake_df   = split_df[split_df["label"] == "fake"].head(per_class)
        split_df  = pd.concat([real_df, fake_df]).reset_index(drop=True)
        log.info(f"  Limited to {len(split_df):,} files ({per_class} real + {per_class} fake)")

    paths  = split_df["path"].tolist()
    labels = [1 if l == "fake" else 0 for l in split_df["label"].tolist()]

    log.info(f"  Extracting {len(paths):,} files with {n_workers} parallel workers...")
    t0 = time.time()

    results = Parallel(n_jobs=n_workers, verbose=0, prefer="threads")(
        delayed(extract_one)(p) for p in paths
    )

    elapsed = time.time() - t0
    log.info(f"  Done in {elapsed:.1f}s  ({elapsed/max(len(paths),1):.2f}s per file)")

    X_list, y_list, failed = [], [], 0
    for feats, label in zip(results, labels):
        if feats is not None:
            X_list.append(feats)
            y_list.append(label)
        else:
            failed += 1

    if not X_list:
        return None, None, failed

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int32)
    log.info(f"  Shape={X.shape}  Real={(y==0).sum()}  Fake={(y==1).sum()}  Failed={failed}")
    return X, y, failed


def main():
    args = parse_args()

    log.info(SEP)
    log.info("Deepfake Audio — Stage 1: FAST Feature Extraction")
    log.info(f"  SR={SR}  Duration={DURATION}s  Workers={args.workers}")
    if args.max_files:
        log.info(f"  Max files per split: {args.max_files} (balanced per class)")
    log.info(SEP)

    if not Path(MANIFEST_CSV).exists():
        log.error("data_manifest.csv not found. Run deepfake_eda.py first.")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_CSV)
    log.info(f"Manifest: {len(df):,} files")

    total_start = time.time()

    for split in df["split"].unique():
        split_df = df[df["split"] == split].reset_index(drop=True)
        log.info(f"\n--- Split: {split} ({len(split_df):,} files) ---")

        X, y, failed = process_split(split_df, split, args.max_files, args.workers)
        if X is None:
            continue

        np.save(f"features_{split}.npy", X)
        np.save(f"labels_{split}.npy", y)
        log.info(f"  Saved features_{split}.npy + labels_{split}.npy")

    log.info(SEP)
    log.info(f"Total time: {time.time()-total_start:.1f}s")
    log.info("Stage 1 COMPLETE")
    log.info(SEP)


if __name__ == "__main__":
    main()
