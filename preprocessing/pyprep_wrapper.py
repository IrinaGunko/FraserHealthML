import mne
from pyprep.prep_pipeline import PrepPipeline
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_pyprep(raw, config):
    sample_rate = raw.info["sfreq"]
    line_freqs = np.arange(config.line_noise, sample_rate / 2, config.line_noise)
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": line_freqs,
        "max_iterations": config.max_iterations
    }
    other_kwargs = {
        "ransac": config.ransac,
        "channel_wise": config.channel_wise,
        "random_state": config.random_state,
        "filter_kwargs": {"method": config.filter_method},
        "matlab_strict": config.matlab_strict
    }
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    prep = PrepPipeline(raw, prep_params, montage, **other_kwargs)
    prep.fit()
    raw = prep.raw
    logger.info("PREP pipeline completed")
    return raw