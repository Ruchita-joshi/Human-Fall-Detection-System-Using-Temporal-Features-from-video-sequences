# app.py
import streamlit as st
import subprocess
import tempfile
import os
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv

# -----------------------------
# Configuration
# -----------------------------
MAIN_SCRIPT = "main_temporal.py"
MODEL_IMPORTANCE_IMG = "temporal_feature_importance.png"
MODEL_PICKLE = "fall_detection_temporal_rf.pkl"
SCALER_PICKLE = "scaler_temporal.pkl"
HISTORY_FILE = "history.csv"

EVAL_SUMMARY = {
    "accuracy": 95.74,
    "precision": 95.81,
    "recall": 93.90,
    "f1": 94.85,
}
CONFUSION_MATRIX = np.array([[1155, 35], [52, 801]])

executor = ThreadPoolExecutor(max_workers=1)

# -----------------------------
# Page config & CSS (black/purple/white)
# -----------------------------
st.set_page_config(page_title="FALLGUARD AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    /* Base */
    .stApp { background: linear-gradient(180deg, #0b0810 0%, #120a19 100%); color: #ffffff; }
    .stMarkdown, .stText, .stButton, .stMetric { color: #ffffff !important; font-size:30px}

    /* Tabs centered & styled */
    .stTabs [data-baseweb="tab-list"] { justify-content: center !important; }
    .stTabs [data-baseweb="tab"] {
        font-size: 20px !important;
        padding: 12px 28px !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #8e5aff !important;
        color: #ffffff !important;
        transform: translateY(-2px);
    }
    .stTabs [aria-selected="true"] {
        background: #8e5aff !important;
        color: #ffffff !important;
        font-weight: 700;
        box-shadow: 0 6px 18px rgba(139,82,255,0.18);
    }

    /* Big Start Button */
    div.stButton > button {
        background: linear-gradient(90deg,#8e5aff,#b36bff) !important;

        color: white !important;
        padding: 16px 20px !important;
        font-size: 20px !important;
        border-radius: 14px !important;
        height:70px;
        
        width: 1000px !important;

    }

    /* Status card */
    .status-card { 
        padding: 20px; 
        border-radius: 16px; 
        text-align: center; 
        font-size: 40px; 
        font-weight: 900; 
        color: white; 
        margin-top: 20px;
    }
    .normal { background: #0f3b17 !important; color: #6dff9a !important; }
    .fall { background: #3b0f0f !important; color: #ff6d6d !important; }

    body, .streamlit-expanderHeader, p, div {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Header
# -----------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    try:
        st.image("/mnt/data/Screenshot 2025-12-12 195759.png", width=80)
    except Exception:
        pass
with col2:
    st.markdown("<h1 style='text-align:center; color:#4B0082; letter-spacing:2px'>FALLGUARD AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#ffffff'>Temporal Analysis for Fall Detection System</p>", unsafe_allow_html=True)
with col3:
    st.write("")

# Readiness check
missing_files = [f for f in (MODEL_PICKLE, SCALER_PICKLE) if not os.path.exists(f)]
if missing_files:
    st.warning(f"Model files missing: {', '.join(missing_files)}. Place pickles in the app directory or train the model.")

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["📤 Upload & Detect", "📊 Model Metrics", "🕑 History"])


# -----------------------------
# Helper: run subprocess
# -----------------------------
def run_subprocess(cmd):
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return proc.returncode, proc.stdout
    except Exception as e:
        return -1, str(e)

# -----------------------------
# Tab 1: Upload & Detect
# -----------------------------
with tabs[0]:

    st.markdown("<h3 style='text-align:center;'>Upload & Detect</h3>", unsafe_allow_html=True)

    left, right = st.columns([3, 1])
    with left:

        # -----------------------------
        # ✔ Only MP4 Allowed
        # -----------------------------
        uploaded = st.file_uploader("Upload MP4 Video Only", type=["mp4"])

        if uploaded is not None and not uploaded.name.lower().endswith(".mp4"):
            st.error("❌ Only MP4 files are allowed.")
            uploaded = None

        video_path = None

        if uploaded is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded.getbuffer())
                tfile.flush()
                video_path = tfile.name

            try:
                st.video(video_path)
            except:
                st.info("Preview not available in this environment.")

            with right:
                st.markdown(f"**{uploaded.name}**")
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                st.markdown(f"{size_mb:.2f} MB")

            if st.button("🚀 Start Analysis"):

                cmd = ["python", MAIN_SCRIPT, "--video", video_path]
                future = executor.submit(run_subprocess, cmd)

                pbar = st.progress(0.0)
                i = 0.0
                while not future.done():
                    time.sleep(0.2)
                    i = min(0.95, i + 0.03)
                    pbar.progress(i)

                retcode, output = future.result()
                pbar.progress(1.0)

                # -----------------------------
                # ✔ Improved Fall Parsing
                # -----------------------------
                frames = None
                falls = 0
                highest_conf = 0.0

                for line in output.splitlines():
                    if not line:
                        continue
                    s = line.strip()
                    lower = s.lower()

                    if lower.startswith("frames processed"):
                        try:
                            frames = int(s.split(":")[-1].strip())
                        except:
                            pass

                    if "fall detected" in lower:
                        falls += 1

                    if "confidence" in lower:
                        try:
                            tail = s.split("confidence:")[-1].strip()
                            if tail.endswith("%"):
                                tail = float(tail[:-1]) / 100.0
                            else:
                                tail = float(tail)
                                if tail > 1:
                                    tail = tail / 100.0
                            if tail > highest_conf:
                                highest_conf = tail
                        except:
                            pass

                # -----------------------------
                # ✔ FALL / NORMAL Status Center Display (fixed)
                # -----------------------------
                if falls > 0:
                    st.markdown("<div class='status-card fall'>FALL DETECTED</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='status-card normal'>NORMAL</div>", unsafe_allow_html=True)

                # Save history
                try:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    row = [uploaded.name, frames if frames else "", falls, f"{highest_conf*100:.1f}%", ts]
                    exists = os.path.exists(HISTORY_FILE)
                    with open(HISTORY_FILE, "a", newline="") as hf:
                        w = csv.writer(hf)
                        if not exists:
                            w.writerow(["Video", "Frames", "Falls", "Highest Confidence", "Timestamp"])
                        w.writerow(row)
                except:
                    pass

                st.download_button("Download Log", data=output, file_name="analysis_log.txt")

                try:
                    os.remove(video_path)
                except:
                    pass

# -----------------------------
# Tab 2 & Tab 3 (unchanged)
# -----------------------------
# Keeping them exactly same as you provided

with tabs[1]:
    st.markdown("<h3 style='text-align:center;'>Model Performance Metrics</h3>", unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    metrics_data = [
        ("Accuracy", 0.9574),
        ("Precision", 0.9581),
        ("Recall", 0.9390),
        ("F1-Score", 0.9485)
    ]

    for col, (label, value) in zip([col_m1, col_m2, col_m3, col_m4], metrics_data):
        with col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value * 100,
                number={'suffix': '%', 'font': {'color': '#ffffff', 'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#ffffff'},
                    'bar': {'color': '#8e5aff'},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 0,
                },
            ))
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#0d0b10")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"<div style='text-align:center; font-weight:700;'>{label}</div>", unsafe_allow_html=True)

    st.divider()

    st.subheader("Confusion Matrix")
    cm = CONFUSION_MATRIX
    fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred Normal", "Pred Fall"], y=["Actual Normal", "Actual Fall"], colorscale='Purples', textfont={"size": 40}))
                                     
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.markdown("<h3 style='text-align:center;'>Analysis History</h3>", unsafe_allow_html=True)

    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(
                HISTORY_FILE,
                names=["Video", "Frames", "Falls", "Highest Confidence", "Timestamp"],
                header=0,
                on_bad_lines="skip"   # 🔥 Skip corrupted rows
            )

            # 🔥 Keep only 5 columns even if an extra value sneaked in
            df = df.iloc[:, :5]

            st.dataframe(df)

        except Exception as e:
            st.error(f"Error reading history file: {e}")

    else:
        st.info("No history recorded yet.")
