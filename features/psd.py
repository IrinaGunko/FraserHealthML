# features/psd.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from mne.time_frequency import psd_array_welch


def compute_psd(signals, config):

    signals = np.asarray(signals)

    # Ensure we have (n_parcels, n_times)
    if signals.ndim != 2:
        raise ValueError(f"signals must be 2D (n_parcels, n_times) or (n_times, n_parcels), got shape {signals.shape}")

    n0, n1 = signals.shape
    if n0 < n1:
        # assume (n_parcels, n_times)
        n_parcels, n_times = n0, n1
        data = signals
    else:
        # assume (n_times, n_parcels)
        n_times, n_parcels = n0, n1
        data = signals.T  # → (n_parcels, n_times)

    sfreq = float(getattr(config, "sfreq", 256))
    fmin = float(getattr(config, "psd_fmin", 1.0))
    fmax = float(getattr(config, "psd_fmax", 49.0))
    n_fft = int(getattr(config, "psd_n_fft", 512))

    psd, freqs = psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        average="mean",
    )

    # Make sure shape is (n_parcels, n_freqs)
    if psd.ndim == 1:
        psd = np.tile(psd, (n_parcels, 1))

    return psd, freqs
