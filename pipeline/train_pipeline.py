"""
Deepfake Audio Detection — Full Training Pipeline (UPDATED)
============================================================
Stage 0 -> EDA & manifest
Stage 1 -> Fast parallel feature extraction (--max_files 3000 --workers 4)
Stage 2 -> Model training (targets Accuracy >= 80%, EER <= 12%, F1 >= 80%)

Usage:
  python train_pipeline.py                        # full run
  python train_pipeline.py --skip_eda             # skip EDA
  python train_pipeline.py --skip_features        # skip feature extraction
  python train_pipeline.py --max_files 4000       # more files = better accuracy
"""

import os
import sys
import json
import logging
import argparse
import warnings
import subprocess
from pathlib import Path

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("deepfake_pipeline.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("Deepfake-Pipeline")
SEP = "=" * 68


def parse_args():
    p = argparse.ArgumentParser(
        description="Deepfake Audio Detection — Full Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root",     default="data",
                   help="Root folder with train/test/eval audio.")
    p.add_argument("--max_files",     type=int, default=3000,
                   help="Max files per split for feature extraction (default: 3000).")
    p.add_argument("--workers",       type=int, default=4,
                   help="Parallel workers for feature extraction (default: 4).")
    p.add_argument("--skip_eda",      action="store_true",
                   help="Skip EDA if data_manifest.csv already exists.")
    p.add_argument("--skip_features", action="store_true",
                   help="Skip feature extraction if .npy files exist.")
    p.add_argument("--skip_train",    action="store_true",
                   help="Skip training if saved_model/ exists.")
    return p.parse_args()


def run_stage(script: str, label: str, extra_args: list = None):
    """Run a pipeline stage script as subprocess."""
    log.info(SEP)
    log.info(f"STARTING  {label}  ->  {script}")
    log.info(SEP)

    if not Path(script).exists():
        raise FileNotFoundError(
            f"Script '{script}' not found. "
            "Make sure all scripts are in the same folder."
        )

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"]       = "1"

    cmd = [sys.executable, "-X", "utf8", script] + (extra_args or [])
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    for line in proc.stdout:
        log.info(f"  [{script}]  {line.rstrip()}")
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            f"Stage '{label}' FAILED (exit code {proc.returncode}). "
            "Check deepfake_pipeline.log for details."
        )
    log.info(f"[OK]  {label} complete.\n")


def validate(path: str, label: str):
    if not Path(path).exists():
        log.error(f"Expected output '{path}' ({label}) not found.")
        sys.exit(1)
    log.info(f"[OK]  Verified: {path}")


def main():
    args = parse_args()

    log.info(SEP)
    log.info("  Deepfake Audio Detection — Full Training Pipeline")
    log.info(SEP)
    log.info(f"  Data root    : {args.data_root}")
    log.info(f"  Max files    : {args.max_files} per split "
             f"({args.max_files//2} real + {args.max_files//2} fake)")
    log.info(f"  Workers      : {args.workers}")
    log.info(f"  Skip EDA     : {args.skip_eda}")
    log.info(f"  Skip features: {args.skip_features}")
    log.info(f"  Skip train   : {args.skip_train}")
    log.info("")

    # Check data folder
    if not Path(args.data_root).exists():
        log.error(
            f"Data folder '{args.data_root}' not found!\n"
            "Download from: https://www.kaggle.com/datasets/"
            "mohammedabdeldayem/the-fake-or-real-dataset\n"
            "Structure needed:\n"
            "  data/train/real/  data/train/fake/\n"
            "  data/test/real/   data/test/fake/\n"
            "  data/eval/real/   data/eval/fake/"
        )
        sys.exit(1)

    # ── Stage 0: EDA ──────────────────────────────────────────────────
    if args.skip_eda and Path("data_manifest.csv").exists():
        log.info("[SKIP]  EDA (data_manifest.csv already exists)")
    else:
        run_stage("deepfake_eda.py", "Stage 0 -- EDA & Manifest")
        validate("data_manifest.csv", "Data manifest")

    # ── Stage 1: Feature Extraction (FAST PARALLEL) ───────────────────
    if args.skip_features and Path("features_train.npy").exists():
        log.info("[SKIP]  Feature extraction (.npy files already exist)")
    else:
        # Pass max_files and workers to the fast feature script
        feat_args = [
            "--max_files", str(args.max_files),
            "--workers",   str(args.workers),
        ]
        run_stage("deepfake_features.py",
                  "Stage 1 -- Fast Parallel Feature Extraction",
                  extra_args=feat_args)
        validate("features_train.npy", "Train features")
        validate("labels_train.npy",   "Train labels")

    # ── Stage 2: Model Training ───────────────────────────────────────
    if args.skip_train and Path("saved_model/deepfake_model.pkl").exists():
        log.info("[SKIP]  Model training (saved_model/ already exists)")
    else:
        run_stage("deepfake_train.py",
                  "Stage 2 -- Model Training & Evaluation")
        validate("saved_model/deepfake_model.pkl", "Trained model")
        validate("deepfake_metrics.json",           "Metrics JSON")
        validate("deepfake_report.txt",             "Performance report")

    # ── Final Summary ─────────────────────────────────────────────────
    log.info(SEP)
    log.info("  PIPELINE COMPLETE")
    log.info(SEP)

    if Path("deepfake_metrics.json").exists():
        with open("deepfake_metrics.json") as f:
            m = json.load(f)

        log.info("")
        log.info("  ── Metric Results ────────────────────────────────")
        log.info(f"  {'Metric':<28} {'Achieved':>8}  {'Required':>8}  Pass?")
        log.info(f"  {'-'*56}")

        acc  = m["overall_accuracy"]
        eer  = m["eer_percent"]
        f1   = m["f1_score_macro"]
        ar   = m["per_class_accuracy_real"]
        af   = m["per_class_accuracy_fake"]

        log.info(f"  {'Overall Accuracy':<28} {acc*100:>7.2f}%  {'>=80%':>8}  {'OK' if acc>=0.80 else 'FAIL'}")
        log.info(f"  {'Equal Error Rate (EER)':<28} {eer:>7.2f}%  {'<=12%':>8}  {'OK' if eer<=12.0 else 'FAIL'}")
        log.info(f"  {'F1 Score (macro)':<28} {f1*100:>7.2f}%  {'>=80%':>8}  {'OK' if f1>=0.80 else 'FAIL'}")
        log.info(f"  {'Per-Class Acc (Real)':<28} {ar*100:>7.2f}%  {'>=75%':>8}  {'OK' if ar>=0.75 else 'FAIL'}")
        log.info(f"  {'Per-Class Acc (Fake)':<28} {af*100:>7.2f}%  {'>=75%':>8}  {'OK' if af>=0.75 else 'FAIL'}")
        log.info(f"  {'-'*56}")

        status = "ALL THRESHOLDS PASSED" if m["all_thresholds_passed"] else "SOME FAILED"
        log.info(f"  Status: {status}")

        if not m["all_thresholds_passed"]:
            log.info("")
            log.info("  [TIP] To improve accuracy, use more files:")
            log.info(f"  python train_pipeline.py --max_files {args.max_files + 1000} --skip_eda")

    log.info("")
    log.info("  Output files:")
    for f in ["data_manifest.csv",
              "features_train.npy", "labels_train.npy",
              "features_test.npy",  "labels_test.npy",
              "deepfake_metrics.json", "deepfake_report.txt",
              "saved_model/deepfake_model.pkl"]:
        exists = Path(f).exists()
        log.info(f"  {'[OK]' if exists else '[X] '}  {f}")

    log.info("")
    log.info("  Launch web app:")
    log.info("    streamlit run app.py")
    log.info(SEP)


if __name__ == "__main__":
    main()
