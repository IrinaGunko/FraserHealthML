import os
import glob
import logging
import re
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import diptest
from specparam import SpectralModel
from features.psd import compute_psd
from config import OUTPUT_DIR, H5_FOLDER, AlphaAnalysisConfig, S_A_AXIS_RANKS
from utils.HDF5Handler import H5FileHandler
from utils.label_utils import parse_label_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_label(label):
    return re.sub(r'(.+)-(lh|rh)$', r'\2_\1', label)

def plot_full_grid(df, metadata):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans
    from scipy.stats import gaussian_kde
    import numpy as np
    import diptest

    subject = metadata.get("subject_id", "unknown")
    session = metadata.get("session", "unknown")
    task = metadata.get("task", "")
    acq = metadata.get("acquisition", "")

    bands = ['alpha']
    cf_cols = ['alpha_peak_cf']
    pw_cols = ['alpha_peak_pw']

    fig, axs = plt.subplots(len(bands), 3, figsize=(16, 5))
    axs = np.atleast_2d(axs)

    for i, (band, cf_col, pw_col) in enumerate(zip(bands, cf_cols, pw_cols)):
        clean_df = df.dropna(subset=[cf_col, pw_col, 'sa_rank']).copy()

        if clean_df.empty:
            continue

        # --- KMeans Clustering ---
        values = clean_df[[cf_col]].values
        kmeans = KMeans(n_clusters=2, random_state=12345)
        clean_df['cluster'] = kmeans.fit_predict(values)

        # --- KDE + Dip ---
        cf = clean_df[cf_col].values
        kde = gaussian_kde(cf)
        xs = np.linspace(min(cf), max(cf), 200)
        dip_stat, pval = diptest.diptest(cf)

        # --- Scatter Plot: CF vs S-A axis ---
        sns.scatterplot(
            data=clean_df,
            x='sa_rank',
            y=cf_col,
            hue='cluster',
            palette='Set2',
            alpha=0.9,
            legend=False,
            ax=axs[i, 0]
        )
        axs[i, 0].set_title(f'{band.capitalize()} CF vs. S-A Axis')
        axs[i, 0].set_xlabel("S-A Axis Rank")
        axs[i, 0].set_ylabel("Center Frequency (Hz)")

        # --- Histogram + KDE ---
        sns.histplot(
            data=clean_df,
            x=cf_col,
            hue='cluster',
            palette='Set2',
            bins=20,
            kde=True,
            element="step",
            ax=axs[i, 1]
        )
        axs[i, 1].set_title(f'{band.capitalize()} CF Histogram')
        axs[i, 1].set_xlabel("Center Frequency (Hz)")
        axs[i, 1].set_ylabel("Count")

        # --- Distribution with KDE and Dip ---
        axs[i, 2].hist(cf, bins=20, density=True, alpha=0.6, label="Histogram")
        axs[i, 2].plot(xs, kde(xs), label="KDE", linewidth=2)
        axs[i, 2].set_title(f"{band.capitalize()} KDE\nDip={dip_stat:.4f}, p={pval:.4f}")
        axs[i, 2].set_xlabel("Center Frequency (Hz)")
        axs[i, 2].set_ylabel("Density")
        axs[i, 2].legend()

    fig.suptitle(
        f"Subject: {subject}, Session: {session}, Task: {task}, Acquisition: {acq}",
        fontsize=14,
        y=1.05
    )
    plt.tight_layout()
    plt.show()


def process_h5_file(h5_path, config: AlphaAnalysisConfig, save_dir):
    signals, label_names, metadata = H5FileHandler.load_hdf5_parcel_signals(h5_path)
    base_name = os.path.splitext(os.path.basename(h5_path))[0]
    output_csv = os.path.join(save_dir, f"{base_name}_features.csv")
    psd, freqs = compute_psd(signals, config)
    results = []
    for i, roi_psd in enumerate(psd):
        roi_label = clean_label(label_names[i])
        parsed = parse_label_name(roi_label)
        network = parsed['network_name']
        parcel = parsed['parcel_name']

        model = SpectralModel(
            peak_width_limits=config.peak_width_limits,
            max_n_peaks=config.max_n_peaks,
            min_peak_height=config.min_peak_height,
            aperiodic_mode=config.aperiodic_mode,
            peak_threshold = config.peak_threshold,
            verbose=False
        )
        try:
            model.fit(freqs, roi_psd)
            ap_params, peak_params, r_squared, fit_error, gauss_params = model.get_results()
            model.print_results()
            model.plot()
            #plt.show()
            row = {
                "label": roi_label,
                "network": network,
                "parcel": parcel,
                "r_squared": r_squared,
                "fit_error": fit_error,
                "aperiodic_offset": ap_params[0],
                "aperiodic_exponent": ap_params[1],
                "n_alpha_peaks": 0
            }
            alpha_peaks = [
                (cf, pw, bw) for cf, pw, bw in peak_params
                if config.alpha_start <= cf <= config.alpha_end
            ]
            alpha_peaks.sort(key=lambda p: p[1], reverse=True)

            if alpha_peaks:
                cf, pw, bw = alpha_peaks[0]
                row["alpha_peak_cf"] = cf
                row["alpha_peak_pw"] = pw
                row["alpha_peak_bw"] = bw
            else:
                row["alpha_peak_cf"] = None
                row["alpha_peak_pw"] = None
                row["alpha_peak_bw"] = None

            row["n_alpha_peaks"] = len(alpha_peaks)

        except Exception as e:
            logger.warning(f"Specparam failed for {roi_label}: {e}")

        row.update(metadata)
        results.append(row)
    df = pd.DataFrame(results)
    sa_df = pd.read_csv(S_A_AXIS_RANKS)
    df = df.merge(sa_df[["label", "averagerank.wholebrain"]], on="label", how="inner")
    df = df.rename(columns={"averagerank.wholebrain": "sa_rank"})
    df = df.drop(columns=[col for col in df.columns if col in ["sampling_rate", "created_on"]], errors="ignore")
    desired_order = [
        "subject_id", "session", "task", "acquisition",
        "sa_rank", "network", "parcel", "label",
        "alpha_peak_cf", "alpha_peak_pw", "alpha_peak_bw"
    ]
    df = df[desired_order + [col for col in df.columns if col not in desired_order]]
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved CSV: {output_csv}")
    # Run diptest on the non-null alpha peak center frequencies
    cf_values = df['alpha_peak_cf'].dropna().values

    if len(cf_values) >= 4:  # Dip test requires at least 4 values
        dip, pval = diptest.diptest(cf_values)
        if pval < 0.05:
            interpretation = "Multimodal distribution likely (reject unimodality)"
        else:
            interpretation = "Unimodal distribution likely (fail to reject unimodality)"
        logger.info(f"Hartigans Dip Test on alpha peak CFs: dip={dip:.4f}, p={pval:.4f} → {interpretation}")

        # Add diptest summary to a metadata row in CSV (optional)
        df["dipstat_alpha_cf"] = dip
        df["dip_pval_alpha_cf"] = pval
    else:
        logger.warning("Too few alpha peak CF values for diptest.")
        df["dipstat_alpha_cf"] = np.nan
        df["dip_pval_alpha_cf"] = np.nan
    plot_full_grid(df, metadata)


def run_batch_alpha_analysis(config: AlphaAnalysisConfig):
    input_dir = H5_FOLDER
    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    h5_files = [f for f in os.listdir(input_dir) if f.endswith(".h5")]
    if not h5_files:
        logger.warning(f"No HDF5 files found in {input_dir}")
        return
    logger.info(f"Found {len(h5_files)} files to process in {input_dir}")
    for h5_file in h5_files:
        h5_path = os.path.join(input_dir, h5_file)
        try:
            process_h5_file(h5_path, config, output_dir)
        except Exception as e:
            logger.error(f"Failed to process {h5_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run alpha peak extraction from HDF5 EEG parcel signals")
    parser.add_argument("--filepath", help="Path to a specific .h5 file")
    parser.add_argument("--index", type=int, help="Index of the .h5 file to process (for SLURM job arrays)")
    parser.add_argument("--data-folder", default=H5_FOLDER, help="Folder with .h5 EEG files")

    args = parser.parse_args()
    config = AlphaAnalysisConfig()

    if args.filepath:
        logger.info(f"Running on single file: {args.filepath}")
        process_h5_file(args.filepath, config, OUTPUT_DIR)

    elif args.index is not None:
        h5_files = sorted(glob.glob(os.path.join(args.data_folder, "*.h5")))
        if args.index >= len(h5_files):
            raise IndexError(f"Index {args.index} out of range: {len(h5_files)} files found in {args.data_folder}")
        filepath = h5_files[args.index]
        logger.info(f"Running SLURM index mode. Processing file: {filepath}")
        process_h5_file(filepath, config, OUTPUT_DIR)

    else:
        logger.info("Running in full-batch mode")
        run_batch_alpha_analysis(config)