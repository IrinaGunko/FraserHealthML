#!/usr/bin/env python3
"""
pipeline/extract_features.py

Feature extraction pipeline for the Streamlit app.
Mirrors fraser_allpeaks_specparam.py (the working cluster script) exactly -
same PSD, same specparam params, same label cleaning, same column order.

Public API
----------
    run_pipeline(h5_path, progress_callback=None, debug=False) -> pd.DataFrame
    debug_hdf5(h5_path)   <- standalone HDF5 inspector, call before run_pipeline
"""

import logging
import os
import re

import h5py
import numpy as np
import pandas as pd
from specparam import SpectralModel

from features.psd import compute_psd
from config import AlphaAnalysisConfig, S_A_AXIS_RANKS
from utils.label_utils import parse_label_name
import specparam as _specparam

logger = logging.getLogger(__name__)

# Detect specparam version once at import time
_SPECPARAM_V2 = int(_specparam.__version__.split('.')[0]) >= 2
logger.debug("specparam version: %s (v2=%s)", _specparam.__version__, _SPECPARAM_V2)


# =============================================================================
# DEBUG HELPERS
# =============================================================================

def debug_hdf5(h5_path: str) -> dict:
    """
    Inspect every dataset in the HDF5 file and return a summary dict.
    Also logs everything — call this before run_pipeline to diagnose issues.

    Returns dict with keys: datasets, signals_shape, label_sample, sfreq_found
    """
    logger.info("=" * 60)
    logger.info("HDF5 DEBUG: %s", h5_path)
    logger.info("=" * 60)

    info = {"path": h5_path, "datasets": {}, "signals_shape": None,
            "label_sample": [], "sfreq_found": None}

    with h5py.File(h5_path, "r") as f:
        logger.info("Top-level keys: %s", list(f.keys()))

        for key in f.keys():
            ds = f[key]
            if hasattr(ds, "shape"):
                dtype = str(ds.dtype)
                shape = ds.shape
                info["datasets"][key] = {"shape": shape, "dtype": dtype}
                logger.info("  [%s]  shape=%s  dtype=%s", key, shape, dtype)

                # Show a sample value for scalars / short arrays
                if ds.size == 1:
                    val = ds[()]
                    logger.info("    value: %s", val)
                    info["sfreq_found"] = val  # might be sfreq
                elif ds.size <= 10:
                    logger.info("    values: %s", ds[:])

        # Specific focus on label_tcs
        if "label_tcs" in f:
            sig = f["label_tcs"][:]
            info["signals_shape"] = sig.shape
            n0, n1 = sig.shape
            logger.info("")
            logger.info("label_tcs shape: %s", sig.shape)
            if n0 < n1:
                logger.info("  -> interpretation: (%d parcels, %d timepoints)  [n_parcels < n_times]", n0, n1)
            else:
                logger.info("  -> interpretation: (%d timepoints, %d parcels)  [n_times < n_parcels — will be transposed]", n0, n1)
            logger.info("  min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
                        sig.min(), sig.max(), sig.mean(), sig.std())
            logger.info("  any NaN: %s  any Inf: %s",
                        np.isnan(sig).any(), np.isinf(sig).any())

        # Specific focus on label_names
        if "label_names" in f:
            raw = f["label_names"][:]
            decoded = [
                x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
                for x in raw
            ]
            info["label_sample"] = decoded[:5]
            logger.info("")
            logger.info("label_names: %d labels", len(decoded))
            logger.info("  first 5: %s", decoded[:5])
            logger.info("  last  5: %s", decoded[-5:])

    logger.info("=" * 60)
    return info


def debug_psd(psd: np.ndarray, freqs: np.ndarray,
              label_names: list, n_samples: int = 5,
              save_path: str | None = None):
    """
    Log PSD diagnostics and optionally save a plot of sample parcels.

    Parameters
    ----------
    psd         : (n_parcels, n_freqs)
    freqs       : (n_freqs,)
    label_names : list of parcel names (already clean_label'd)
    n_samples   : how many parcels to plot
    save_path   : if set, saves PNG to this path
    """
    logger.info("")
    logger.info("PSD DEBUG")
    logger.info("  psd shape  : %s  (should be n_parcels x n_freqs)", psd.shape)
    logger.info("  freqs shape: %s", freqs.shape)
    logger.info("  freq range : %.2f - %.2f Hz", freqs.min(), freqs.max())
    logger.info("  freq resolution: %.4f Hz", freqs[1] - freqs[0])
    logger.info("  psd min=%.6f  max=%.6f  mean=%.6f", psd.min(), psd.max(), psd.mean())
    logger.info("  any NaN: %s  any Inf: %s",
                np.isnan(psd).any(), np.isinf(psd).any())

    # Check alpha band specifically (7-14 Hz)
    alpha_mask = (freqs >= 7) & (freqs <= 14)
    alpha_psd  = psd[:, alpha_mask]
    logger.info("  alpha band (7-14 Hz): %d freq bins", alpha_mask.sum())
    logger.info("  alpha psd mean=%.6f  max=%.6f", alpha_psd.mean(), alpha_psd.max())

    # Find parcels with strongest alpha power (sanity check)
    alpha_power = alpha_psd.mean(axis=1)
    top5_idx    = np.argsort(alpha_power)[-5:][::-1]
    logger.info("  top 5 parcels by mean alpha power:")
    for idx in top5_idx:
        name = label_names[idx] if idx < len(label_names) else f"parcel_{idx}"
        logger.info("    [%d] %s  alpha_power=%.6f", idx, name, alpha_power[idx])

    # Plot
    if save_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            indices = np.linspace(0, len(psd) - 1, min(n_samples, len(psd)),
                                  dtype=int)
            fig, axes = plt.subplots(n_samples, 1,
                                     figsize=(10, 3 * n_samples),
                                     sharex=True)
            if n_samples == 1:
                axes = [axes]

            for ax, idx in zip(axes, indices):
                name = label_names[idx] if idx < len(label_names) else f"parcel_{idx}"
                ax.semilogy(freqs, psd[idx], color="#2166ac", lw=1.5)
                ax.axvspan(7, 14, alpha=0.15, color="orange", label="alpha (7-14 Hz)")
                ax.set_ylabel("Power")
                ax.set_title(f"Parcel {idx}: {name}", fontsize=9)
                ax.legend(fontsize=7, frameon=False)
                ax.spines[["top", "right"]].set_visible(False)

            axes[-1].set_xlabel("Frequency (Hz)")
            fig.suptitle("PSD Debug — sample parcels", fontsize=11, y=1.01)
            plt.tight_layout()
            plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            logger.info("  PSD plot saved: %s", save_path)
        except Exception as e:
            logger.warning("  PSD plot failed: %s", e)


# =============================================================================
# LABEL HELPERS
# =============================================================================

def clean_label(label: str) -> str:
    """'17Networks_LH_ContA_Cingm_1-lh'  ->  'lh_17Networks_LH_ContA_Cingm_1'"""
    return re.sub(r'(.+)-(lh|rh)$', r'\2_\1', label)


# =============================================================================
# HDF5 LOADER
# =============================================================================

def load_fraser_hdf5(h5_path: str):
    """
    Load a Fraser Health beamformed HDF5 file.
    Returns: signals (n_parcels, n_times), label_names, metadata dict.
    """
    with h5py.File(h5_path, "r") as f:
        signals         = f["label_tcs"][:]
        raw_label_names = f["label_names"][:]

    label_names = [
        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in raw_label_names
    ]

    base_name  = os.path.basename(h5_path)
    stem       = base_name.replace(".hdf5", "").replace(".h5", "")
    # The UUID is the leading part of the filename before -ico-4-ltc
    # e.g. '575f3167-871c-4be8-a12f-48ceb16915ca-ico-4-ltc' -> subject_id = UUID
    subject_id = stem.split("-ico-")[0] if "-ico-" in stem else stem

    metadata = {"subject_id": subject_id}
    return signals, label_names, metadata


# =============================================================================
# COLUMN ORDER
# =============================================================================

_DESIRED_ORDER = [
    "subject_id",
    "sa_rank", "network", "parcel", "label",
    "alpha_peak_index", "alpha_peak_cf", "alpha_peak_pw", "alpha_peak_bw",
    "aperiodic_offset", "aperiodic_exponent", "r_squared", "fit_error",
]


# =============================================================================
# PUBLIC API
# =============================================================================

def run_pipeline(
    h5_path,
    config=None,
    sa_csv_path=None,
    save_csv_path=None,
    progress_callback=None,
    debug=False,
    debug_plot_path=None,
):
    """
    Run the full feature extraction pipeline on one Fraser Health HDF5 file.

    Parameters
    ----------
    h5_path           : path to .hdf5 file
    config            : AlphaAnalysisConfig instance (uses defaults if None)
    sa_csv_path       : override path to S-A axis ranks CSV
    save_csv_path     : if set, saves result CSV here
    progress_callback : callable(current: int, total: int) for progress bar
    debug             : if True, logs HDF5 structure and PSD diagnostics
    debug_plot_path   : if set (e.g. "/tmp/psd_debug.png"), saves PSD plot here

    Returns
    -------
    pd.DataFrame  - one row per (parcel x alpha peak), empty if none found
    """
    if config is None:
        config = AlphaAnalysisConfig()

    # Default: look for schaefer400_sa_axis_merged.csv in S-A_Axis/ subfolder
    if sa_csv_path:
        sa_path = sa_csv_path
    else:
        # Try schaefer400_sa_axis_merged.csv first, fall back to S_A_AXIS_RANKS from config
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _new  = os.path.join(_base, "S-A_Axis", "schaefer400_sa_axis_merged.csv")
        sa_path = _new if os.path.exists(_new) else S_A_AXIS_RANKS

    # -- Load -----------------------------------------------------------------
    if debug:
        debug_hdf5(h5_path)

    signals, label_names, metadata = load_fraser_hdf5(h5_path)
    n_parcels = len(label_names)

    if debug:
        logger.info("After load_fraser_hdf5:")
        logger.info("  signals.shape : %s", signals.shape)
        logger.info("  n_parcels     : %d", n_parcels)
        logger.info("  config.sfreq  : %s", config.sfreq)

    # -- PSD ------------------------------------------------------------------
    logger.info("Computing PSD for %d parcels ...", n_parcels)
    psd, freqs = compute_psd(signals, config)

    if debug:
        clean_names = [clean_label(l) for l in label_names]
        debug_psd(psd, freqs, clean_names, n_samples=5,
                  save_path=debug_plot_path)

    # -- Specparam per parcel -------------------------------------------------
    logger.info("Fitting SpectralModel (alpha %d-%d Hz) ...",
                config.alpha_start, config.alpha_end)

    all_rows      = []
    n_no_peaks    = 0
    n_no_alpha    = 0
    n_fit_failed  = 0

    for i, roi_psd in enumerate(psd):
        roi_label = clean_label(label_names[i])
        parsed    = parse_label_name(roi_label)
        network   = parsed["network_name"]
        parcel    = parsed["parcel_name"]

        model = SpectralModel(
            peak_width_limits = config.peak_width_limits,
            max_n_peaks       = config.max_n_peaks,
            min_peak_height   = config.min_peak_height,
            aperiodic_mode    = config.aperiodic_mode,
            peak_threshold    = config.peak_threshold,
            verbose           = False,
        )

        try:
            if _SPECPARAM_V2:
                # specparam v2: fit requires freq_range; results on sub-object
                model.fit(freqs, roi_psd, [freqs[0], freqs[-1]])
                ap_params   = model.results.params.aperiodic.params
                peak_params = model.results.params.periodic.params
                r_squared   = model.results.metrics.results['gof_rsquared']
                fit_error   = model.results.metrics.results['error_mae']
            else:
                # specparam v1 / fooof
                model.fit(freqs, roi_psd)
                ap_params, peak_params, r_squared, fit_error, _ = model.get_results()
        except Exception as e:
            n_fit_failed += 1
            if n_fit_failed <= 3:  # always log first 3 errors regardless of debug flag
                logger.warning("  [%d] %s — fit FAILED: %s: %s",
                               i, roi_label, type(e).__name__, e)
            elif debug:
                logger.warning("  [%d] %s — fit FAILED: %s", i, roi_label, e)
            if progress_callback:
                progress_callback(i + 1, n_parcels)
            continue

        # Normalise peak_params to list of (cf, pw, bw) tuples.
        # v1 get_results() returns a list of tuples or empty list.
        # v2 peak_params_ returns a 2D ndarray (n_peaks, 3) or shape (0, 3).
        if peak_params is None:
            peak_list = []
        elif isinstance(peak_params, np.ndarray):
            peak_list = [tuple(row) for row in peak_params]  # (n_peaks, 3) -> list of tuples
        else:
            peak_list = list(peak_params)

        if len(peak_list) == 0:
            n_no_peaks += 1
            if progress_callback:
                progress_callback(i + 1, n_parcels)
            continue

        alpha_peaks = [
            (cf, pw, bw)
            for cf, pw, bw in peak_list
            if config.alpha_start <= cf <= config.alpha_end
        ]

        if not alpha_peaks:
            n_no_alpha += 1
            if debug and i < 10:
                all_cfs = [cf for cf, pw, bw in peak_list]
                logger.info("  [%d] %s — peaks found but NONE in alpha range %d-%d Hz. "
                            "All peak CFs: %s",
                            i, roi_label,
                            config.alpha_start, config.alpha_end,
                            [f"{cf:.2f}" for cf in all_cfs])
            if progress_callback:
                progress_callback(i + 1, n_parcels)
            continue

        for alpha_index, (cf, pw, bw) in enumerate(alpha_peaks, start=1):
            row = {
                "label":              roi_label,
                "network":            network,
                "parcel":             parcel,
                "alpha_peak_index":   alpha_index,
                "alpha_peak_cf":      cf,
                "alpha_peak_pw":      pw,
                "alpha_peak_bw":      bw,
                "aperiodic_offset":   ap_params[0] if len(ap_params) > 0 else np.nan,
                "aperiodic_exponent": ap_params[1] if len(ap_params) > 1 else np.nan,
                "r_squared":          r_squared,
                "fit_error":          fit_error,
            }
            row.update(metadata)
            all_rows.append(row)

        if progress_callback:
            progress_callback(i + 1, n_parcels)

    # -- Specparam summary ----------------------------------------------------
    logger.info("Specparam summary:")
    logger.info("  total parcels    : %d", n_parcels)
    logger.info("  fit failed       : %d", n_fit_failed)
    logger.info("  no peaks at all  : %d", n_no_peaks)
    logger.info("  peaks outside alpha range: %d", n_no_alpha)
    logger.info("  rows collected   : %d", len(all_rows))

    if not all_rows:
        logger.warning("No alpha peaks found in %s", h5_path)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # -- Gradient / S-A axis merge -------------------------------------------
    # Uses schaefer400_sa_axis_merged.csv — merges ALL gradient columns by label
    if os.path.exists(sa_path):
        sa_df = pd.read_csv(sa_path)
        # Merge all gradient columns (everything except row_index and name which are redundant)
        gradient_cols = [c for c in sa_df.columns if c not in ("row_index", "name")]
        df = df.merge(sa_df[gradient_cols], on="label", how="left")
        n_matched = df["averagerank.wholebrain"].notna().sum()
        logger.info("Gradient merge: %d/%d rows matched", n_matched, len(df))
    else:
        logger.warning("Gradient CSV not found: %s", sa_path)

    # -- Column ordering ------------------------------------------------------
    ordered = [c for c in _DESIRED_ORDER if c in df.columns]
    rest    = [c for c in df.columns if c not in _DESIRED_ORDER]
    df      = df[ordered + rest]

    # -- Optional save --------------------------------------------------------
    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
        df.to_csv(save_csv_path, index=False)
        logger.info("Saved: %s", save_csv_path)

    logger.info("Done - %d rows (%d parcels with alpha peaks)",
                len(df), df["label"].nunique())
    return df