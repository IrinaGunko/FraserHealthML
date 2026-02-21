import mne
import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess_raw(raw, config):
    raw.notch_filter(freqs=config.line_noise, picks=['eeg'], method='iir')
    logger.info(f"Notch filter applied at {config.line_noise} Hz")

    raw.filter(l_freq=config.lfreq, h_freq=config.hfreq, picks='eeg', method='iir')
    logger.info(f"Band-pass filter applied from {config.lfreq} to {config.hfreq} Hz")

    if raw.info['sfreq'] != config.target_freq:
        raw.resample(config.target_freq)
        logger.info(f"Resampled to {config.target_freq} Hz")
    return raw
