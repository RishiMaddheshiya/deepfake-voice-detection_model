"""
Deepfake Audio Detection — Stage 0: EDA & Data Exploration
===========================================================
Dataset : The Fake-or-Real Dataset (Kaggle)
          kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset

Expected folder structure:
    data/
    └── LA/
        ├── train/
        │   ├── real/   ← genuine human speech (.wav / .flac)
        │   └── fake/   ← AI-generated deepfake speech
        └── test/
            ├── real/
            └── fake/

Output: data_manifest.csv  (path, label, split, duration_sec, sr)
"""

import os
import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("deepfake_eda.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("Deepfake-EDA")
SEP = "=" * 60

# ── CONFIG ────────────────────────────────────────────────────────────
DATA_ROOT   = Path("data")            # adjust if your folder is elsewhere
OUTPUT_CSV  = "data_manifest.csv"
AUDIO_EXTS  = {".wav", ".flac", ".mp3", ".ogg"}


def find_audio_files(root: Path):
    """Walk the data directory and collect all audio file paths + labels."""
    records = []
    for split in ["train", "test", "eval"]:
        split_dir = root / split
        if not split_dir.exists():
            # Also try without 'LA' subfolder
            split_dir = root / "LA" / split
        if not split_dir.exists():
            continue

        for label in ["real", "fake", "genuine", "spoof"]:
            label_dir = split_dir / label
            if not label_dir.exists():
                continue

            # Normalize label
            norm_label = "real" if label in ("real", "genuine") else "fake"

            for ext in AUDIO_EXTS:
                for fpath in label_dir.glob(f"**/*{ext}"):
                    records.append({
                        "path"  : str(fpath),
                        "label" : norm_label,
                        "split" : split,
                    })

    return records


def get_audio_info(fpath: str):
    """Get duration and sample rate without loading the whole file."""
    try:
        import librosa
        duration = librosa.get_duration(path=fpath)
        y, sr    = librosa.load(fpath, sr=None, duration=0.1)
        return round(duration, 3), int(sr)
    except Exception:
        return None, None


def main():
    log.info(SEP)
    log.info("Deepfake Audio Detection — Stage 0: EDA")
    log.info(SEP)

    # ── Try to import librosa ────────────────────────────────────────
    try:
        import librosa
        log.info("librosa available")
    except ImportError:
        log.error("librosa not found. Install: pip install librosa soundfile")
        sys.exit(1)

    # ── Find data ────────────────────────────────────────────────────
    log.info(f"\nSearching for audio files under: {DATA_ROOT.resolve()}")
    if not DATA_ROOT.exists():
        log.error(
            f"Data folder '{DATA_ROOT}' not found!\n"
            "Download from: https://www.kaggle.com/datasets/"
            "mohammedabdeldayem/the-fake-or-real-dataset\n"
            "Extract to 'data/' folder in project root."
        )
        sys.exit(1)

    records = find_audio_files(DATA_ROOT)
    if not records:
        log.error(
            "No audio files found! Check folder structure:\n"
            "  data/train/real/*.wav\n"
            "  data/train/fake/*.wav\n"
            "  data/test/real/*.wav\n"
            "  data/test/fake/*.wav"
        )
        sys.exit(1)

    log.info(f"Found {len(records):,} audio files total")

    # ── Build manifest ───────────────────────────────────────────────
    log.info("\nBuilding data manifest (reading durations)...")
    df = pd.DataFrame(records)

    durations = []
    sample_rates = []
    for i, row in df.iterrows():
        dur, sr = get_audio_info(row["path"])
        durations.append(dur)
        sample_rates.append(sr)
        if (i + 1) % 500 == 0:
            log.info(f"  Processed {i+1:,} / {len(df):,} files...")

    df["duration_sec"] = durations
    df["sample_rate"]  = sample_rates
    df = df.dropna(subset=["duration_sec"])

    # ── EDA Report ───────────────────────────────────────────────────
    log.info(SEP)
    log.info("EDA REPORT")
    log.info(SEP)

    log.info(f"\nTotal files      : {len(df):,}")
    log.info(f"Real (Genuine)   : {(df['label']=='real').sum():,}")
    log.info(f"Fake (Deepfake)  : {(df['label']=='fake').sum():,}")

    log.info("\nSplit distribution:")
    log.info(df.groupby(["split","label"]).size().to_string())

    log.info("\nDuration stats (seconds):")
    log.info(df["duration_sec"].describe().round(3).to_string())

    log.info("\nSample rate distribution:")
    log.info(df["sample_rate"].value_counts().to_string())

    if df["duration_sec"].notna().any():
        short = (df["duration_sec"] < 1.0).sum()
        long  = (df["duration_sec"] > 10.0).sum()
        log.info(f"\nFiles < 1 sec  : {short:,}")
        log.info(f"Files > 10 sec : {long:,}")

    # ── Save manifest ────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"\nSaved manifest -> '{OUTPUT_CSV}'  ({len(df):,} rows)")
    log.info(SEP)
    log.info("Stage 0 COMPLETE — Ready for feature extraction")
    log.info(SEP)


if __name__ == "__main__":
    main()
