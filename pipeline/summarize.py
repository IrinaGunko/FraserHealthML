#!/usr/bin/env python3
"""
pipeline/summarize.py

Stage 3: collapse per-ROI CSV (N rows) into a single summary row
that matches the feature vector expected by the XGBoost models.

Mirrors build_file1_row() from the training summarization script exactly —
same stats, same network groupings, same column names.

Public API
----------
    summarize(df_roi: pd.DataFrame) -> pd.DataFrame  # 1-row DataFrame
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Network mappings (identical to training script) ──────────────────────────

NETWORK_TO_MAJOR = {
    'Default A':                    'DefaultMode',
    'Default B':                    'DefaultMode',
    'Default C':                    'DefaultMode',
    'Control A':                    'ExecutiveControl',
    'Control B':                    'ExecutiveControl',
    'Control C':                    'ExecutiveControl',
    'Somatomotor A':                'Somatomotor',
    'Somatomotor B':                'Somatomotor',
    'Visual A':                     'Visual',
    'Visual B':                     'Visual',
    'Limbic A':                     'Limbic',
    'Limbic B':                     'Limbic',
    'Salience/Ventral Attention A': 'Salience/VentralAttention',
    'Salience/Ventral Attention B': 'Salience/VentralAttention',
    'Dorsal Attention A':           'DorsalAttention',
    'Dorsal Attention B':           'DorsalAttention',
    'Temporal Parietal':            'Temporoparietal',
}

ALL_17_NETWORKS = list(NETWORK_TO_MAJOR.keys())
ALL_8_MAJOR     = list(dict.fromkeys(NETWORK_TO_MAJOR.values()))

# ── Feature groups (identical to training script) ────────────────────────────

ALPHA_FEATURES     = ['alpha_peak_cf', 'alpha_peak_pw', 'alpha_peak_bw']
APERIODIC_FEATURES = ['aperiodic_offset', 'aperiodic_exponent', 'r_squared', 'fit_error']
GRADIENT_FEATURES  = [
    'T1T2ratio', 'G1.fMRI', 'Evolution.Expansion', 'AllometricScaling.PNC20mm',
    'PET.AG', 'CBF', 'PC1.AHBA', 'PC1.Neurosynth', 'BigBrain.Histology', 'Cortical.Thickness',
]
RANK_FEATURES = [
    'averagerank.wholebrain', 'finalrank.wholebrain',
    'averagerank.hemisphere',  'finalrank.hemisphere',
]

PER_NETWORK_FEATURES = (
    ALPHA_FEATURES
    + ['aperiodic_offset', 'aperiodic_exponent']
    + GRADIENT_FEATURES
    + ['averagerank.wholebrain', 'finalrank.wholebrain']
)

SUBJECT_COLS = [
    'subject_id', 'Hashed_ReportURN', 'Focal_Epi', 'Gen_Epi', 'Focal_Non_epi',
    'Gen_Non_epi', 'Abnormality', 'Birthdate', 'Sex', 'Age',
]


# ── Stats helpers (identical to training script) ─────────────────────────────

def _summary_stats(series: pd.Series, prefix: str) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {
            f"{prefix}_mean":        np.nan,
            f"{prefix}_median":      np.nan,
            f"{prefix}_p5":          np.nan,
            f"{prefix}_p95":         np.nan,
            f"{prefix}_range_5_95":  np.nan,
        }
    p5, p95 = np.percentile(s, [5, 95])
    return {
        f"{prefix}_mean":        s.mean(),
        f"{prefix}_median":      s.median(),
        f"{prefix}_p5":          p5,
        f"{prefix}_p95":         p95,
        f"{prefix}_range_5_95":  p95 - p5,
    }


def _global_feature_summaries(df: pd.DataFrame) -> dict:
    row = {}
    for feat in ALPHA_FEATURES + APERIODIC_FEATURES + GRADIENT_FEATURES + RANK_FEATURES:
        if feat in df.columns:
            row.update(_summary_stats(df[feat], prefix=feat))
    return row


def _network_counts_and_pcts(df: pd.DataFrame, network_col: str,
                              all_networks: list, prefix: str) -> dict:
    total  = len(df)
    counts = df[network_col].value_counts()
    row = {}
    for net in all_networks:
        safe = net.replace('/', '_').replace(' ', '_')
        cnt  = int(counts.get(net, 0))
        pct  = cnt / total if total > 0 else np.nan
        row[f"{prefix}_{safe}_count"] = cnt
        row[f"{prefix}_{safe}_pct"]   = pct
    return row


def _per_network_feature_summaries(df: pd.DataFrame, network_col: str,
                                    all_networks: list, prefix: str) -> dict:
    row = {}
    for net in all_networks:
        safe   = net.replace('/', '_').replace(' ', '_')
        subset = df[df[network_col] == net]
        for feat in PER_NETWORK_FEATURES:
            if feat in df.columns:
                row.update(_summary_stats(subset[feat],
                                          prefix=f"{prefix}_{safe}_{feat}"))
    return row


def _run_diptest(series: pd.Series) -> dict:
    vals = series.dropna().to_numpy()
    if len(vals) >= 4:
        try:
            import diptest
            dip_stat, pval = diptest.diptest(vals)
            return {'dipstat_alpha_cf': dip_stat, 'dip_pval_alpha_cf': pval}
        except ImportError:
            logger.warning("diptest not installed — dip test skipped. "
                           "Run: pip install diptest")
    return {'dipstat_alpha_cf': np.nan, 'dip_pval_alpha_cf': np.nan}


def _subject_identifiers(df: pd.DataFrame) -> dict:
    first = df.iloc[0]
    return {col: first[col] if col in df.columns else np.nan
            for col in SUBJECT_COLS}


# ── Public API ────────────────────────────────────────────────────────────────

def summarize(df_roi: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a per-ROI DataFrame into a single summary row.

    Parameters
    ----------
    df_roi : per-ROI DataFrame as produced by run_pipeline()
             (N rows, one per parcel × alpha peak, already gradient-merged)

    Returns
    -------
    pd.DataFrame with exactly 1 row — the feature vector for XGBoost inference.
    """
    # Filter to dominant peak only (alpha_peak_index == 1)
    if 'alpha_peak_index' in df_roi.columns:
        df = df_roi[df_roi['alpha_peak_index'] == 1].copy()
        n_dropped = len(df_roi) - len(df)
        if n_dropped > 0:
            logger.info("Summarize: kept %d/%d rows (alpha_peak_index==1)",
                        len(df), len(df_roi))
    else:
        df = df_roi.copy()
        logger.warning("alpha_peak_index column not found — using all rows")

    if len(df) == 0:
        logger.warning("No rows with alpha_peak_index==1 — returning empty DataFrame")
        return pd.DataFrame()

    row = {}

    # 1. Subject identifiers
    row.update(_subject_identifiers(df))

    # 2. Dip test on alpha CF
    row.update(_run_diptest(df['alpha_peak_cf']))

    # 3. Global feature summaries
    row.update(_global_feature_summaries(df))

    # 4. Add major network column
    df['major_network'] = df['network'].map(NETWORK_TO_MAJOR)
    n_unmapped = df['major_network'].isna().sum()
    if n_unmapped > 0:
        logger.warning("%d parcels had unrecognized network names", n_unmapped)

    # 5. 17-network counts + pcts
    row.update(_network_counts_and_pcts(df, 'network',       ALL_17_NETWORKS, 'net17'))

    # 6. 8-major network counts + pcts
    row.update(_network_counts_and_pcts(df, 'major_network', ALL_8_MAJOR,     'net8'))

    # 7. Per-network feature summaries — 17 networks
    row.update(_per_network_feature_summaries(df, 'network',       ALL_17_NETWORKS, 'net17'))

    # 8. Per-network feature summaries — 8 major networks
    row.update(_per_network_feature_summaries(df, 'major_network', ALL_8_MAJOR,     'net8'))

    result = pd.DataFrame([row])
    logger.info("Summarize: produced 1 row x %d columns", len(result.columns))
    return result