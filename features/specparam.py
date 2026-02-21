

import numpy as np
from specparam import SpectralModel
from config import AlphaAnalysisConfig
from specparam.data.periodic import get_band_peak

def extract_specparam(freqs, psd, config: AlphaAnalysisConfig):
    psd = np.where(psd <= 0, 1e-12, psd)

    model = SpectralModel(
        peak_width_limits=config.peak_width_limits,
        max_n_peaks=config.max_n_peaks,
        min_peak_height=config.min_peak_height,
        peak_threshold=config.peak_threshold,
        aperiodic_mode=config.aperiodic_mode
    )

    try:
        model.fit(freqs, psd)
        #print_specparam_summary(model)
    except Exception as e:
        return _return_empty_specparam(config.bands)

    offset, slope = model.aperiodic_params_
    features = {
        'aperiodic_offset': offset,
        'aperiodic_slope': slope,
        'r_squared': model.r_squared_,
        'error': model.error_,
        'n_peaks': len(model.peak_params_),
    }

    for band_name, (fmin, fmax) in config.bands.items():
        peak = get_band_peak(model, (fmin, fmax))
        cf, pw, bw = peak if peak is not None else (np.nan, np.nan, np.nan)
        features[f'{band_name}_cf'] = cf
        features[f'{band_name}_pw'] = pw
        features[f'{band_name}_bw'] = bw

    return features



def _return_empty_specparam(bands):
    features = {
        'aperiodic_offset': np.nan,
        'aperiodic_slope': np.nan,
        'r_squared': np.nan,
        'error': np.nan,
        'n_peaks': 0,
    }
    for name in bands:
        features[f'{name}_cf'] = np.nan
        features[f'{name}_pw'] = np.nan
        features[f'{name}_bw'] = np.nan
    return features
def print_specparam_summary(model):
    print(f"Fit frequency range: {model.freqs[0]:.2f}–{model.freqs[-1]:.2f} Hz")
    print(f"Aperiodic params (offset, slope): {model.aperiodic_params_}")
    print(f"Number of peaks: {len(model.peak_params_)}")
    print(f"R²: {model.r_squared_:.4f}, Error: {model.error_:.4e}")
    for i, (cf, pw, bw) in enumerate(model.peak_params_):
        print(f"  Peak {i+1}: CF={cf:.2f}, PW={pw:.2f}, BW={bw:.2f}")