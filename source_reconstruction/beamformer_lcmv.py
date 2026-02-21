import mne

def apply_lcmv_beamformer(raw, fwd):

    print("🌀 Using LCMV MNE beamformer...")

    # Compute noise covariance from the raw EEG data
    print("📏 Computing noise covariance...")
    noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, method="auto")

    # Regularize noise covariance
    noise_cov_reg = mne.cov.regularize(
        noise_cov, raw.info, mag=0.05, grad=0.05, eeg=0.1
    )

    # Create LCMV beamformer filters
    print("📡 Creating LCMV filters...")
    filters = mne.beamformer.make_lcmv(
        raw.info,
        fwd,
        noise_cov,
        reg=0.05,
        pick_ori="max-power",
        weight_norm="unit-noise-gain",
        rank=None
    )

    # Apply LCMV beamformer to the raw EEG data
    print("📊 Applying LCMV beamformer...")
    stc = mne.beamformer.apply_lcmv_raw(raw, filters)

    print("✅ LCMV beamformer computation complete.")
    return stc
