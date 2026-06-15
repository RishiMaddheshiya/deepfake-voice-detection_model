"""
Deepfake Audio Detection — Streamlit Web App
=============================================
Upload an audio file → Get prediction (Genuine / Deepfake) + confidence score
"""

import io
import json
import pickle
import tempfile
import warnings
from pathlib import Path

import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #080C14 0%, #0D1421 50%, #080C14 100%);
    color: #E2E8F0;
}
.stApp::before {
    content: '';
    position: fixed; top:0; left:0; right:0; bottom:0;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none; z-index:0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050810 0%, #0A0F1E 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.15) !important;
}
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0);    }
}
.main .block-container { animation: fadeInUp 0.5s ease both; }

.stButton > button {
    background: linear-gradient(135deg, #7B2FFF 0%, #00D4FF 100%) !important;
    border: none !important; border-radius: 12px !important;
    color: white !important; font-weight: 600 !important;
    font-size: 16px !important; padding: 14px 28px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(123,47,255,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(123,47,255,0.55) !important;
}

/* Genuine banner */
@keyframes genuinePulse {
    0%,100% { box-shadow: 0 0 20px rgba(0,255,136,0.3); }
    50%      { box-shadow: 0 0 40px rgba(0,255,136,0.6); }
}
.genuine-banner {
    background: linear-gradient(135deg, rgba(0,255,136,0.12), rgba(0,212,255,0.08));
    border: 1px solid rgba(0,255,136,0.5);
    border-radius: 16px; padding: 24px; margin: 16px 0;
    animation: fadeInUp 0.4s ease, genuinePulse 2.5s ease-in-out infinite;
}
.genuine-banner h2 { color: #00FF88; margin: 0; font-size: 26px; }

/* Deepfake banner */
@keyframes deepfakePulse {
    0%,100% { box-shadow: 0 0 20px rgba(255,51,102,0.4); }
    50%      { box-shadow: 0 0 45px rgba(255,51,102,0.7); }
}
.deepfake-banner {
    background: linear-gradient(135deg, rgba(255,51,102,0.15), rgba(123,47,255,0.10));
    border: 1px solid rgba(255,51,102,0.6);
    border-radius: 16px; padding: 24px; margin: 16px 0;
    animation: fadeInUp 0.4s ease, deepfakePulse 2.5s ease-in-out infinite;
}
.deepfake-banner h2 { color: #FF3366; margin: 0; font-size: 26px; }

[data-testid="stMetric"] {
    background: rgba(13,20,40,0.7) !important;
    border: 1px solid rgba(0,212,255,0.12) !important;
    border-radius: 14px !important; padding: 20px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(0,212,255,0.35) !important;
    transform: translateY(-4px) !important;
}
[data-testid="stMetricValue"] { color: #00D4FF !important; font-weight:700 !important; }

.conf-bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 50px; height: 10px; margin: 10px 0; overflow: hidden;
}
@keyframes barGrow { from { width: 0%; } }
.conf-bar-fill {
    height: 100%; border-radius: 50px;
    animation: barGrow 0.8s cubic-bezier(0.34,1.56,0.64,1) both;
}

[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,212,255,0.3) !important;
    border-radius: 14px !important;
    background: rgba(0,212,255,0.03) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,255,0.6) !important;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#7B2FFF, #00D4FF); border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("saved_model/deepfake_model.pkl")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def extract_features(fpath: str, arts: dict) -> np.ndarray:
    """
    Extract 284 features matching the trained model exactly.
    MFCC(40) + delta + delta-delta + Mel + Chroma + Spectral = 284 features
    """
    import librosa

    SR       = arts.get("sr", 16000)
    DURATION = arts.get("duration", 3.0)
    HOP      = 512
    N_MFCC   = 40
    N_MELS   = 64

    y, _ = librosa.load(fpath, sr=SR, duration=DURATION, mono=True)
    tlen  = int(SR * DURATION)
    y     = np.pad(y, (0, max(0, tlen - len(y))), mode="constant")[:tlen]

    f = []

    # MFCC + delta + delta-delta (240 features)
    mfcc    = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP)
    mfcc_d  = librosa.feature.delta(mfcc)
    mfcc_dd = librosa.feature.delta(mfcc, order=2)
    for i in range(N_MFCC):
        f.extend([mfcc[i].mean(), mfcc[i].std()])
        f.extend([mfcc_d[i].mean(), mfcc_d[i].std()])
        f.extend([mfcc_dd[i].mean(), mfcc_dd[i].std()])

    # Mel Spectrogram stats (4 features)
    mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    f.extend([mel_db.mean(), mel_db.std(), mel_db.max(), mel_db.min()])

    # Chroma (24 features)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR, hop_length=HOP)
    for i in range(12):
        f.extend([chroma[i].mean(), chroma[i].std()])

    # Spectral features (10 features)
    f.extend([
        librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP).mean(),
        librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP).std(),
        librosa.feature.spectral_bandwidth(y=y, sr=SR, hop_length=HOP).mean(),
        librosa.feature.spectral_bandwidth(y=y, sr=SR, hop_length=HOP).std(),
        librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=HOP).mean(),
        librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=HOP).std(),
        librosa.feature.spectral_contrast(y=y, sr=SR, hop_length=HOP).mean(),
        librosa.feature.spectral_contrast(y=y, sr=SR, hop_length=HOP).std(),
        librosa.feature.spectral_flatness(y=y, hop_length=HOP).mean(),
        librosa.feature.spectral_flatness(y=y, hop_length=HOP).std(),
    ])

    # ZCR + RMS (4 features)
    f.extend([
        librosa.feature.zero_crossing_rate(y, hop_length=HOP).mean(),
        librosa.feature.zero_crossing_rate(y, hop_length=HOP).std(),
        librosa.feature.rms(y=y, hop_length=HOP).mean(),
        librosa.feature.rms(y=y, hop_length=HOP).std(),
    ])

    # Tonnetz (2 features)
    try:
        h = librosa.effects.harmonic(y)
        t = librosa.feature.tonnetz(y=h, sr=SR)
        f.extend([t.mean(), t.std()])
    except Exception:
        f.extend([0.0, 0.0])

    return np.array(f, dtype=np.float32)


def predict_audio(fpath: str, arts: dict) -> dict:
    """Run full inference pipeline: features -> scaler -> selector -> ensemble -> iso -> result."""
    feats = extract_features(fpath, arts)
    X     = arts["scaler"].transform(feats.reshape(1, -1))

    # Feature selection
    if arts.get("selector") is not None:
        X = arts["selector"].transform(X)

    # Get probabilities from all classifiers
    clfs  = arts.get("clfs", {})
    probs = []

    # Use clfs dict if available (new model format)
    if clfs:
        for name, clf in clfs.items():
            try:
                probs.append(clf.predict_proba(X)[0][1])
            except Exception:
                pass
    else:
        # Fallback: backward compat keys
        for key in ["clf_rf","clf_gb","clf_et","clf_lr","clf_xgb"]:
            clf = arts.get(key)
            if clf is not None:
                try:
                    probs.append(clf.predict_proba(X)[0][1])
                except Exception:
                    pass

    if not probs:
        prob_fake = 0.5
    else:
        raw_prob = float(np.mean(probs))
        # Apply isotonic calibration if available
        iso = arts.get("iso")
        if iso is not None:
            prob_fake = float(iso.predict([raw_prob])[0])
        else:
            prob_fake = raw_prob

    threshold = arts.get("threshold", 0.5)
    is_fake   = bool(prob_fake >= threshold)

    return {
        "is_fake"       : is_fake,
        "prob_fake"     : round(prob_fake, 4),
        "prob_real"     : round(1 - prob_fake, 4),
        "confidence_pct": round(float(prob_fake if is_fake else 1 - prob_fake) * 100, 1),
        "label"         : "Deepfake (AI-Generated)" if is_fake else "Genuine (Human)",
        "rf_score"      : round(float(probs[0]) if len(probs) > 0 else 0.5, 3),
        "gb_score"      : round(float(probs[1]) if len(probs) > 1 else 0.5, 3),
        "lr_score"      : round(float(probs[-1]) if probs else 0.5, 3),
    }


# ── SIDEBAR ──────────────────────────────────────────────────────────
arts = load_model()

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px;'>
      <div style='font-size:52px;'>🎙️</div>
      <div style='font-size:20px; font-weight:700; background:linear-gradient(135deg,#E2E8F0,#00D4FF);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-top:8px;'>
        Deepfake Detector
      </div>
      <div style='color:#64748B; font-size:11px; letter-spacing:2px; text-transform:uppercase; margin-top:4px;'>
        Audio Authentication
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if arts:
        st.success("✅ Model Loaded")
        if Path("deepfake_metrics.json").exists():
            with open("deepfake_metrics.json") as f:
                m = json.load(f)
            st.markdown(f"""
            <div style='background:rgba(0,212,255,0.05); border:1px solid rgba(0,212,255,0.15);
                 border-radius:10px; padding:12px; font-size:12px; color:#94A3B8;'>
              <b style='color:#00D4FF;'>Model Performance</b><br><br>
              Accuracy : <b style='color:#00FF88;'>{m['overall_accuracy']*100:.1f}%</b><br>
              EER      : <b style='color:#00FF88;'>{m['eer_percent']:.1f}%</b><br>
              F1 Score : <b style='color:#00FF88;'>{m['f1_score_macro']*100:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Model not found!\nRun: `python train_pipeline.py`")

    st.divider()
    st.markdown("""
    <div style='color:#334155; font-size:11px; text-align:center;'>
      MARS Open Projects 2026<br>Deepfake Audio Detection
    </div>
    """, unsafe_allow_html=True)


# ── MAIN PAGE ────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:0 0 24px;'>
  <h1 style='font-size:32px; font-weight:700;
       background:linear-gradient(135deg,#E2E8F0,#00D4FF);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;'>
    🎙️ Deepfake Audio Detector
  </h1>
  <p style='color:#64748B; margin:6px 0 0; font-size:15px;'>
    Upload a speech recording to detect whether it is genuine human speech
    or AI-generated deepfake audio.
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Upload zone
uploaded_file = st.file_uploader(
    "Drop an audio file here (.wav / .flac / .mp3)",
    type=["wav", "flac", "mp3", "ogg"],
    help="The model analyzes MFCC, Mel Spectrogram, Chroma and Spectral features."
)

if uploaded_file:
    st.markdown("**Preview:**")
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    st.markdown(f"<div style='color:#64748B; font-size:12px; margin-top:4px;'>File: {uploaded_file.name} | Size: {uploaded_file.size/1024:.1f} KB</div>", unsafe_allow_html=True)

    if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
        if not arts:
            st.error("❌ Model not loaded. Run `python train_pipeline.py` first.")
        else:
            with st.spinner("Extracting audio features and running inference..."):
                try:
                    # Save to temp file
                    suffix = "." + uploaded_file.name.split(".")[-1]
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    result = predict_audio(tmp_path, arts)

                    # ── Result banner ────────────────────────────────
                    if result["is_fake"]:
                        st.markdown(f"""
                        <div class='deepfake-banner'>
                          <h2>🚨 DEEPFAKE DETECTED</h2>
                          <p style='color:#CBD5E1; margin:8px 0 0; font-size:14px;'>
                            This audio recording appears to be <b style='color:#FF3366;'>
                            AI-Generated</b> speech.
                            The system detected synthetic audio artifacts inconsistent
                            with genuine human speech patterns.
                          </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='genuine-banner'>
                          <h2>✅ GENUINE SPEECH</h2>
                          <p style='color:#CBD5E1; margin:8px 0 0; font-size:14px;'>
                            This audio recording appears to be <b style='color:#00FF88;'>
                            Human (Genuine)</b> speech.
                            No significant deepfake artifacts were detected.
                          </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Metrics ──────────────────────────────────────
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Prediction",  result["label"].split(" ")[0])
                    c2.metric("Confidence",  f"{result['confidence_pct']}%")
                    c3.metric("P(Fake)",     f"{result['prob_fake']*100:.1f}%")
                    c4.metric("P(Real)",     f"{result['prob_real']*100:.1f}%")

                    # ── Confidence bar ───────────────────────────────
                    bar_color = "#FF3366" if result["is_fake"] else "#00FF88"
                    bar_w     = int(result["confidence_pct"])
                    st.markdown(f"""
                    <div style='margin:16px 0 4px; color:#64748B; font-size:13px;'>
                      Confidence Score</div>
                    <div class='conf-bar-wrap'>
                      <div class='conf-bar-fill'
                           style='width:{bar_w}%; background:linear-gradient(90deg,{bar_color},{bar_color}88);'>
                      </div>
                    </div>
                    <div style='text-align:right; font-size:12px; color:#64748B;'>
                      {result['confidence_pct']}%</div>
                    """, unsafe_allow_html=True)

                    # ── Model breakdown ──────────────────────────────
                    st.markdown("---")
                    st.markdown("### 📊 Model Breakdown")
                    st.caption("Individual classifier scores (ensemble soft-vote)")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Random Forest",      f"{result['rf_score']*100:.1f}% fake")
                    col2.metric("Gradient Boosting",  f"{result['gb_score']*100:.1f}% fake")
                    col3.metric("Logistic Regression",f"{result['lr_score']*100:.1f}% fake")

                    # ── Feature summary ──────────────────────────────
                    st.markdown("### 🔬 Features Analyzed")
                    st.markdown("""
                    <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px;'>
                    """ + "".join([
                        f"<div style='background:rgba(0,212,255,0.05); border:1px solid rgba(0,212,255,0.1); border-radius:8px; padding:10px 14px; font-size:13px; color:#CBD5E1;'>{name}</div>"
                        for name in [
                            "MFCC (40 coefficients)",
                            "MFCC Delta + Delta-Delta",
                            "Mel Spectrogram (64 bins)",
                            "Chroma Features (12)",
                            "Spectral Centroid",
                            "Spectral Bandwidth",
                            "Spectral Rolloff",
                            "Spectral Contrast",
                            "Zero Crossing Rate",
                            "RMS Energy",
                            "Spectral Flatness",
                            "Tonnetz",
                        ]
                    ]) + "</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Make sure the audio file is a valid .wav or .flac file.")

# ── Model info when no file uploaded ─────────────────────────────────
if not uploaded_file:
    st.markdown("""
    <div style='background:rgba(13,20,40,0.7); border:1px solid rgba(0,212,255,0.1);
         border-radius:16px; padding:32px; text-align:center; margin-top:20px;'>
      <div style='font-size:48px; margin-bottom:16px;'>🎤</div>
      <div style='color:#94A3B8; font-size:16px; margin-bottom:8px;'>
        Upload an audio file to begin detection
      </div>
      <div style='color:#475569; font-size:13px;'>
        Supported formats: WAV · FLAC · MP3 · OGG
      </div>
    </div>
    """, unsafe_allow_html=True)

    if Path("deepfake_metrics.json").exists():
        with open("deepfake_metrics.json") as f:
            m = json.load(f)
        st.markdown("---")
        st.markdown("### 📈 Model Performance Report")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Overall Accuracy",   f"{m['overall_accuracy']*100:.1f}%", "req ≥80%")
        c2.metric("Equal Error Rate",   f"{m['eer_percent']:.1f}%",          "req ≤12%")
        c3.metric("F1 Score",           f"{m['f1_score_macro']*100:.1f}%",   "req ≥80%")
        c4.metric("Acc (Real)",         f"{m['per_class_accuracy_real']*100:.1f}%","req ≥75%")
        c5.metric("Acc (Fake)",         f"{m['per_class_accuracy_fake']*100:.1f}%","req ≥75%")
