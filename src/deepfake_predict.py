"""
Deepfake Audio Detection — Inference Script
============================================
Test the trained model on new audio files.

Usage:
  # Single file
  python deepfake_predict.py --input audio.wav

  # Folder of files
  python deepfake_predict.py --input audio_folder/

  # With output CSV
  python deepfake_predict.py --input audio_folder/ --output results.csv

Output per file:
  - Prediction : Genuine (Human) | Deepfake (AI-Generated)
  - Confidence : probability score (0-100%)
  - EER Score  : distance from decision boundary
"""

import sys
import json
import pickle
import logging
import argparse
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
        logging.FileHandler("deepfake_predict.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("Deepfake-Predict")
SEP = "=" * 60

MODEL_PATH  = Path("saved_model/deepfake_model.pkl")
AUDIO_EXTS  = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Deepfake Audio Detector — Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  required=True,
                   help="Path to audio file OR folder containing audio files.")
    p.add_argument("--output", default="predictions.csv",
                   help="Path for output CSV results.")
    p.add_argument("--model",  default=str(MODEL_PATH),
                   help="Path to saved model pickle.")
    return p.parse_args()


def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        log.error(
            f"Model not found at '{path}'.\n"
            "Run train_pipeline.py first to train the model."
        )
        sys.exit(1)
    with open(path, "rb") as f:
        arts = pickle.load(f)
    log.info(f"Model loaded from '{path}'")
    return arts


def extract_features_single(fpath: str, sr: int = 16000,
                             duration: float = 4.0) -> np.ndarray:
    """Extract features from a single audio file."""
    import librosa

    y, _ = librosa.load(fpath, sr=sr, duration=duration, mono=True)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]

    hop = 512
    features = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=hop)
    for i in range(40):
        features.extend([mfcc[i].mean(), mfcc[i].std()])
    mfcc_d  = librosa.feature.delta(mfcc)
    mfcc_dd = librosa.feature.delta(mfcc, order=2)
    for i in range(40):
        features.extend([mfcc_d[i].mean(), mfcc_d[i].std()])
    for i in range(40):
        features.extend([mfcc_dd[i].mean(), mfcc_dd[i].std()])

    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=hop)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend([mel_db.mean(), mel_db.std(), mel_db.max(), mel_db.min()])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    for i in range(12):
        features.extend([chroma[i].mean(), chroma[i].std()])

    features.extend([
        librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop).mean(),
        librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop).std(),
        librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop).mean(),
        librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop).std(),
        librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop).mean(),
        librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop).std(),
        librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop).mean(),
        librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop).std(),
        librosa.feature.spectral_flatness(y=y, hop_length=hop).mean(),
        librosa.feature.spectral_flatness(y=y, hop_length=hop).std(),
        librosa.feature.zero_crossing_rate(y, hop_length=hop).mean(),
        librosa.feature.zero_crossing_rate(y, hop_length=hop).std(),
        librosa.feature.rms(y=y, hop_length=hop).mean(),
        librosa.feature.rms(y=y, hop_length=hop).std(),
    ])

    try:
        harmonic = librosa.effects.harmonic(y)
        t = librosa.feature.tonnetz(y=harmonic, sr=sr)
        features.extend([t.mean(), t.std()])
    except Exception:
        features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


def predict_single(fpath: str, arts: dict) -> dict:
    """Run inference on a single audio file."""
    try:
        import librosa
        feats = extract_features_single(
            fpath,
            sr=arts.get("sr", 16000),
            duration=arts.get("duration", 4.0)
        )
        X = arts["scaler"].transform(feats.reshape(1, -1))

        p_rf = arts["clf_rf"].predict_proba(X)[0][1]
        p_gb = arts["clf_gb"].predict_proba(X)[0][1]
        p_lr = arts["clf_lr"].predict_proba(X)[0][1]
        prob_fake = (p_rf + p_gb + p_lr) / 3

        threshold = arts.get("threshold", 0.5)
        is_fake   = prob_fake >= threshold

        return {
            "file"           : Path(fpath).name,
            "path"           : str(fpath),
            "prediction"     : "Deepfake (AI-Generated)" if is_fake else "Genuine (Human)",
            "label"          : 1 if is_fake else 0,
            "confidence_pct" : round(float(prob_fake if is_fake else 1 - prob_fake) * 100, 2),
            "prob_fake"      : round(float(prob_fake), 4),
            "prob_real"      : round(float(1 - prob_fake), 4),
            "error"          : None,
        }
    except Exception as e:
        return {
            "file"           : Path(fpath).name,
            "path"           : str(fpath),
            "prediction"     : "Error",
            "label"          : -1,
            "confidence_pct" : 0.0,
            "prob_fake"      : 0.0,
            "prob_real"      : 0.0,
            "error"          : str(e),
        }


def main():
    args = parse_args()

    log.info(SEP)
    log.info("Deepfake Audio Detection — Inference")
    log.info(SEP)
    log.info(f"  Input  : {args.input}")
    log.info(f"  Output : {args.output}")
    log.info(f"  Model  : {args.model}")

    arts = load_model(args.model)
    threshold = arts.get("threshold", 0.5)
    log.info(f"  Decision threshold: {threshold:.2f}")

    # Collect files
    input_path = Path(args.input)
    if input_path.is_file():
        audio_files = [input_path]
    elif input_path.is_dir():
        audio_files = [
            p for ext in AUDIO_EXTS
            for p in input_path.glob(f"**/*{ext}")
        ]
        log.info(f"\nFound {len(audio_files):,} audio files in '{input_path}'")
    else:
        log.error(f"Input path not found: '{input_path}'")
        sys.exit(1)

    if not audio_files:
        log.error("No audio files found.")
        sys.exit(1)

    # Run inference
    log.info(f"\nRunning inference on {len(audio_files):,} files...\n")
    results = []
    n_fake  = 0
    n_real  = 0
    n_err   = 0

    for i, fpath in enumerate(audio_files):
        result = predict_single(str(fpath), arts)
        results.append(result)

        if result["error"]:
            n_err += 1
        elif result["label"] == 1:
            n_fake += 1
        else:
            n_real += 1

        icon = "FAKE" if result["label"] == 1 else ("REAL" if result["label"] == 0 else "ERR ")
        log.info(f"  [{icon}] {result['file']:<40} "
                 f"conf={result['confidence_pct']:5.1f}%  "
                 f"prob_fake={result['prob_fake']:.3f}")

    # Summary
    log.info(f"\n{SEP}")
    log.info(f"  RESULTS SUMMARY")
    log.info(f"  Total files   : {len(results):,}")
    log.info(f"  Genuine (Real): {n_real:,}  ({n_real/len(results)*100:.1f}%)")
    log.info(f"  Deepfake      : {n_fake:,}  ({n_fake/len(results)*100:.1f}%)")
    log.info(f"  Errors        : {n_err:,}")
    log.info(SEP)

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    log.info(f"Results saved -> '{args.output}'")
    log.info("COMPLETE")


if __name__ == "__main__":
    main()
