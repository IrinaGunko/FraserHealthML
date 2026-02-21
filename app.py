#!/usr/bin/env python3
"""
app.py  —  EEG Feature Extraction Demo
Streamlit app: user picks a demo HDF5 or uploads their own,
pipeline extracts per-ROI features, result CSV is downloadable.

Run:
    streamlit run app.py
"""

import io
import logging
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.extract_features import run_pipeline
from pipeline.summarize import summarize
from pipeline.predict import load_models, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# DEMO FILE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

@st.cache_resource
def get_model_registry():
    if not MODELS_DIR.exists():
        return None
    return load_models(str(MODELS_DIR))

# Score interpretation: 1=normal, 2=borderline, 3=probable, 4=definite
# Folder name order: Focal_Epi _ Gen_Epi _ Focal_Non_epi _ Gen_Non_epi _ Abnormality
DEMO_FILES = [
    {
        "label":       "Normal EEG",
        "path":        BASE_DIR / "hdf" / "normal" / "51d51b4e-da45-44d4-922f-b628b33c240a-ico-4-ltc.hdf5",
        "description": "All targets score 1 — no abnormality detected",
        "scores": {
            "Abnormality":    1,
            "Focal_Epi":      1,
            "Focal_Non_epi":  1,
            "Gen_Epi":        1,
            "Gen_Non_epi":    1,
        },
    },
    {
        "label":       "Mild Abnormality (1_1_2_2_2)",
        "path":        BASE_DIR / "hdf" / "1_1_2_2_2" / "000dee81-8ae4-4275-bfdf-556658a2709f-ico-4-ltc.hdf5",
        "description": "Borderline focal/generalised non-epileptic features, mild overall abnormality",
        "scores": {
            "Abnormality":    2,
            "Focal_Epi":      1,
            "Focal_Non_epi":  2,
            "Gen_Epi":        1,
            "Gen_Non_epi":    2,
        },
    },
    {
        "label":       "Focal Epileptic + Generalised Non-Epi (3_1_1_4_4)",
        "path":        BASE_DIR / "hdf" / "3_1_1_4_4" / "381da026-5cab-48b1-bf58-a02704659c68-ico-4-ltc.hdf5",
        "description": "Probable focal epileptic, definite generalised non-epileptic, definite abnormality",
        "scores": {
            "Abnormality":    4,
            "Focal_Epi":      3,
            "Focal_Non_epi":  1,
            "Gen_Epi":        1,
            "Gen_Non_epi":    4,
        },
    },
    {
        "label":       "Definite Focal Epileptic + Abnormal (4_1_1_1_4)",
        "path":        BASE_DIR / "hdf" / "4_1_1_1_4" / "575f3167-871c-4be8-a12f-48ceb16915ca-ico-4-ltc.hdf5",
        "description": "Definite focal epileptic features, definite overall abnormality",
        "scores": {
            "Abnormality":    4,
            "Focal_Epi":      4,
            "Focal_Non_epi":  1,
            "Gen_Epi":        1,
            "Gen_Non_epi":    1,
        },
    },
]

SCORE_LABELS = {1: "Normal", 2: "Borderline", 3: "Probable", 4: "Definite"}
SCORE_COLORS = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}

TARGETS = ["Abnormality", "Focal_Epi", "Focal_Non_epi", "Gen_Epi", "Gen_Non_epi"]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EEG Feature Extraction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 EEG Feature Extraction Pipeline")
st.markdown(
    "Extract alpha-band and aperiodic features from beamformed EEG parcellations "
    "(Fraser Health / Schaefer 400 atlas). "
    "Select a demo recording or upload your own `.hdf5` file."
)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — file selection
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("📂 Select Input File")
    source = st.radio(
        "Source",
        ["Use demo file", "Upload my own .hdf5"],
        index=0,
    )

    h5_path     = None
    demo_scores = None
    file_label  = None

    if source == "Use demo file":
        demo_labels = [d["label"] for d in DEMO_FILES]
        selected_idx = st.selectbox(
            "Choose a demo recording",
            range(len(demo_labels)),
            format_func=lambda i: demo_labels[i],
        )
        demo        = DEMO_FILES[selected_idx]
        h5_path     = str(demo["path"])
        demo_scores = demo["scores"]
        file_label  = demo["label"]

        st.markdown(f"**{demo['label']}**")
        st.caption(demo["description"])
        st.markdown("**Ground truth scores:**")
        for target in TARGETS:
            score = demo_scores[target]
            st.markdown(
                f"{SCORE_COLORS[score]} **{target}**: "
                f"{score} — {SCORE_LABELS[score]}"
            )

        if not Path(h5_path).exists():
            st.error(f"Demo file not found:\n`{h5_path}`")
            h5_path = None

    else:
        uploaded = st.file_uploader(
            "Upload a beamformed .hdf5 file",
            type=["hdf5", "h5"],
            help="Fraser Health format: must contain 'label_tcs' and 'label_names' datasets.",
        )
        if uploaded:
            # Write to a temp file so h5py can open it by path
            tmp = tempfile.NamedTemporaryFile(
                suffix=".hdf5", delete=False)
            tmp.write(uploaded.read())
            tmp.flush()
            tmp.close()
            h5_path    = tmp.name
            file_label = uploaded.name
            st.success(f"Uploaded: **{uploaded.name}**")

    st.divider()
    run_btn = st.button(
        "▶ Process File",
        disabled=(h5_path is None),
        width='stretch',
        type="primary",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL — results
# ══════════════════════════════════════════════════════════════════════════════

if not run_btn:
    st.info("👈 Select a file in the sidebar and click **Process File**.")
    st.stop()

# ── Progress bar ──────────────────────────────────────────────────────────────
st.subheader(f"Processing: {file_label}")
progress_bar  = st.progress(0, text="Starting pipeline…")
status_text   = st.empty()
start_time    = time.time()

def update_progress(current: int, total: int):
    pct  = current / total
    elapsed = time.time() - start_time
    eta  = (elapsed / current * (total - current)) if current > 0 else 0
    progress_bar.progress(pct, text=f"Parcel {current}/{total} — ETA {eta:.0f}s")
    status_text.caption(f"Fitting SpectralModel… {current}/{total} parcels")

# ── Run pipeline ──────────────────────────────────────────────────────────────
with st.spinner("Running specparam on all parcels…"):
    try:
        df = run_pipeline(
            h5_path=h5_path,
            progress_callback=update_progress,
        )
        elapsed = time.time() - start_time
        progress_bar.progress(1.0, text=f"Done in {elapsed:.1f}s")
        status_text.empty()
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Pipeline failed: {e}")
        logger.exception("Pipeline error")
        st.stop()

# ── Empty result guard ────────────────────────────────────────────────────────
if df.empty:
    st.warning("Pipeline completed but no alpha peaks were found in this recording.")
    st.stop()

st.success(
    f"✅ Extracted **{len(df):,} rows** "
    f"({df['label'].nunique()} parcels with alpha peaks) "
    f"in {elapsed:.1f}s"
)
st.divider()

# ── Stage 3: Summarization ────────────────────────────────────────────────────
st.subheader("🔢 Summary Feature Vector")
st.caption("Collapses per-ROI rows into a single row for XGBoost inference "
           "(dominant alpha peak per parcel, global + per-network stats).")

with st.spinner("Building summary feature vector..."):
    try:
        df_summary = summarize(df)
        n_cols = len(df_summary.columns)
        n_nan  = df_summary.isna().sum().sum()
        st.success(
            f"✅ Summary vector: **1 row × {n_cols} features** "
            f"({n_nan} NaN values — expected for missing subject metadata)"
        )

        # Quick breakdown
        col_a, col_b, col_c = st.columns(3)
        global_feat_cols  = [c for c in df_summary.columns
                             if any(c.startswith(p) for p in
                                    ['alpha_peak','aperiodic','T1T2','G1.fMRI',
                                     'Evolution','Allometric','PET','CBF','PC1',
                                     'BigBrain','Cortical','averagerank','finalrank'])
                             and not c.startswith('net')]
        net17_cols = [c for c in df_summary.columns if c.startswith('net17_')]
        net8_cols  = [c for c in df_summary.columns if c.startswith('net8_')]
        col_a.metric("Global features",    len(global_feat_cols))
        col_b.metric("Net-17 features",    len(net17_cols))
        col_c.metric("Net-8 features",     len(net8_cols))

        # Download summary CSV
        summary_csv   = df_summary.to_csv(index=False).encode("utf-8")
        subject_id_s  = df["subject_id"].iloc[0] if "subject_id" in df.columns else "unknown"
        st.download_button(
            label="⬇️ Download Summary Feature Vector CSV",
            data=summary_csv,
            file_name=f"{subject_id_s}_summary_vector.csv",
            mime="text/csv",
            width='stretch',
        )
    except Exception as e:
        st.error(f"Summarization failed: {e}")
        import traceback
        st.code(traceback.format_exc())

st.divider()

# ── Stage 4: Model Predictions ────────────────────────────────────────────────
st.subheader("🤖 XGBoost Model Predictions")

registry = get_model_registry()
if registry is None:
    st.warning("⚠️ No models found — expected at `models/` in the project folder.")
elif df_summary is not None and not df_summary.empty:
    with st.spinner("Running 10 models..."):
        df_preds = predict(df_summary, registry)

    # Display as a styled table grouped by target
    TARGETS_ORDER = ["Abnormality", "Focal_Epi", "Focal_Non_epi", "Gen_Epi", "Gen_Non_epi"]
    TARGET_LABELS = {
        "Abnormality":   "Overall Abnormality",
        "Focal_Epi":     "Focal Epileptic",
        "Focal_Non_epi": "Focal Non-Epileptic",
        "Gen_Epi":       "Generalised Epileptic",
        "Gen_Non_epi":   "Generalised Non-Epileptic",
    }

    cols = st.columns(len(TARGETS_ORDER))
    for col, target in zip(cols, TARGETS_ORDER):
        col.markdown(f"**{TARGET_LABELS[target]}**")
        for _, row in df_preds[df_preds["target"] == target].iterrows():
            variant_label = "Strict (1 vs 4)" if row["variant"] == "strict" else "Relaxed (1-2 vs 3-4)"
            prob = row["probability"]
            thr  = row["threshold"]
            verdict = row["verdict"]
            prob_pct = f"{prob*100:.1f}%" if not (isinstance(prob, float) and prob != prob) else "N/A"
            col.markdown(
                "\n\n".join([
                    f"*{variant_label}*",
                    verdict,
                    f"P = **{prob_pct}** (thr={thr:.2f})",
                ])
            )
            col.divider()

    # Ground truth comparison for demo files
    if demo_scores:
        st.markdown("**Ground truth vs predictions:**")
        rows_compare = []
        for target in TARGETS_ORDER:
            gt_score = demo_scores.get(target, "?")
            for _, prow in df_preds[df_preds["target"] == target].iterrows():
                rows_compare.append({
                    "Target":       TARGET_LABELS[target],
                    "Variant":      prow["variant"],
                    "Ground Truth": f"{gt_score} — {SCORE_LABELS.get(gt_score, '?')}",
                    "Probability":  f"{prow['probability']*100:.1f}%" if prow['probability'] == prow['probability'] else "N/A",
                    "Threshold":    f"{prow['threshold']:.2f}",
                    "Verdict":      prow["verdict"],
                })
        st.dataframe(pd.DataFrame(rows_compare), width="stretch", hide_index=True)