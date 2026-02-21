

import numpy as np
import pandas as pd
from bycycle.features import compute_features
from config import AlphaAnalysisConfig

def extract_burst_features(signal: np.ndarray, sfreq: float, config: AlphaAnalysisConfig) -> dict:
    results = {}

    signal = np.asarray(signal)

    for band_name, (f_low, f_high) in config.bands.items():
        try:
            df_cycles = compute_features(
                signal,
                fs=sfreq,
                f_range=(f_low, f_high),
                burst_method='cycles',
                burst_kwargs={
                    'amplitude_fraction_threshold': 0.3,
                    'amplitude_consistency_threshold': 0.4,
                    'period_consistency_threshold': 0.5,
                    'monotonicity_threshold': 0.8,
                    'min_n_cycles': 3
                },
                center_extrema='amplitude',
                return_samples=False
            )

            if df_cycles.empty:
                raise ValueError("No cycles found")

            bursts_only = df_cycles[df_cycles['is_burst']]

            results[f'{band_name}_burst_rate'] = bursts_only.shape[0] / df_cycles.shape[0]
            results[f'{band_name}_burst_mean_amp'] = bursts_only['volt_amp'].mean()
            results[f'{band_name}_burst_mean_sym'] = bursts_only['time_rdsym'].mean()
            results[f'{band_name}_burst_mean_period'] = bursts_only['period'].mean()
            results[f'{band_name}_burst_n_cycles'] = df_cycles.shape[0]
            results[f'{band_name}_burst_n_bursts'] = bursts_only.shape[0]

        except Exception:
            results[f'{band_name}_burst_rate'] = np.nan
            results[f'{band_name}_burst_mean_amp'] = np.nan
            results[f'{band_name}_burst_mean_sym'] = np.nan
            results[f'{band_name}_burst_mean_period'] = np.nan
            results[f'{band_name}_burst_n_cycles'] = 0
            results[f'{band_name}_burst_n_bursts'] = 0

    return results

