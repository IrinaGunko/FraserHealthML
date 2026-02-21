#!/usr/bin/env python3


import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGETS   = ["Abnormality", "Focal_Epi", "Focal_Non_epi", "Gen_Epi", "Gen_Non_epi"]
VARIANTS  = ["strict", "relaxed"]

SCORE_LABELS = {1: "Normal", 2: "Borderline", 3: "Probable", 4: "Definite"}

# Threshold → verdict mapping
# Model outputs probability of being abnormal/positive class
# Threshold is tuned per model during training


def load_models(models_dir: str) -> dict:
    """
    Load all 10 models + metadata from models_dir.

    Returns
    -------
    dict with keys:
        'models'     : {model_name: XGBClassifier}
        'features'   : {model_name: [col1, col2, ...]}
        'thresholds' : {model_name: float}
        'medians'    : {model_name: {col: median_value}}
    """
    registry = {}

    # feature_columns.json
    feat_path = os.path.join(models_dir, "feature_columns.json")
    with open(feat_path) as f:
        registry["features"] = json.load(f)
    logger.info("Loaded feature columns for %d models", len(registry["features"]))

    # thresholds.json
    thresh_path = os.path.join(models_dir, "thresholds.json")
    with open(thresh_path) as f:
        registry["thresholds"] = json.load(f)

    # train_medians.json
    medians_path = os.path.join(models_dir, "train_medians.json")
    with open(medians_path) as f:
        registry["medians"] = json.load(f)

    # Load .joblib models
    registry["models"] = {}
    for target in TARGETS:
        for variant in VARIANTS:
            name = f"{target}_{variant}"
            path = os.path.join(models_dir, f"{name}.joblib")
            if os.path.exists(path):
                registry["models"][name] = joblib.load(path)
                logger.info("  Loaded: %s", name)
            else:
                logger.warning("  Missing model file: %s", path)

    logger.info("Loaded %d/%d models", len(registry["models"]), len(TARGETS) * len(VARIANTS))
    return registry


def predict(df_summary: pd.DataFrame, registry: dict) -> pd.DataFrame:
    rows = []

    # Flatten summary row to a plain dict once — avoids repeated .values[0] lookups
    summary_row = df_summary.iloc[0].to_dict()

    for target in TARGETS:
        for variant in VARIANTS:
            name = f"{target}_{variant}"

            if name not in registry["models"]:
                rows.append({
                    "target":      target,
                    "variant":     variant,
                    "probability": np.nan,
                    "threshold":   np.nan,
                    "verdict":     "Model not found",
                    "label_map":   "—",
                })
                continue

            model      = registry["models"][name]
            feat_cols  = registry["features"].get(name, [])
            threshold  = registry["thresholds"].get(name, 0.5)
            medians    = registry["medians"].get(name, {})

            # Build the feature dict in one pass, then construct DataFrame once
            # (avoids the PerformanceWarning from repeated column inserts)
            feature_dict = {}
            for col in feat_cols:
                val = summary_row.get(col, medians.get(col, 0.0))
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = medians.get(col, 0.0)
                    logger.debug("  %s: feature '%s' missing/NaN — using median %.4f",
                                 name, col, val)
                feature_dict[col] = val

            X = pd.DataFrame([feature_dict], columns=feat_cols).astype(float)

            try:
                prob     = float(model.predict_proba(X)[0, 1])
                positive = prob >= threshold
                verdict  = "⚠️ Abnormal" if positive else "✅ Normal"
            except Exception as e:
                logger.warning("Prediction failed for %s: %s", name, e)
                prob    = np.nan
                verdict = f"Error: {e}"

            rows.append({
                "target":      target,
                "variant":     variant,
                "probability": round(prob, 4) if not np.isnan(prob) else np.nan,
                "threshold":   round(threshold, 4),
                "verdict":     verdict,
            })

    return pd.DataFrame(rows)