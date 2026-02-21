import os
import mne
import numpy as np
from utils.HDF5Handler import H5FileHandler
from config import SUBJECTS_DIR, OUTPUT_DIR, DEFAULT_SUBJECT, SRC_ICO5

def save_parcel_signals_h5(parcel_signals, bem_name, parc, mode):
    parcel_fname = os.path.join(OUTPUT_DIR, f"{DEFAULT_SUBJECT}-{parc}-{bem_name}-{mode}-parcel_signals.h5")
    data_dict = {"parcel_signals": parcel_signals}
    attr_dict = {
        "subject": DEFAULT_SUBJECT,
        "parcellation": parc,
        "mode": mode
    }
    H5FileHandler.save_h5_file(parcel_fname, data_dict, attr_dict, overwrite=False)

def load_parcel_signals_h5(parc, mode, bem_name):
    parcel_fname = os.path.join(OUTPUT_DIR, f"{DEFAULT_SUBJECT}-{parc}-{bem_name}-{mode}-parcel_signals.h5")
    if os.path.exists(parcel_fname):
        data_dict, attr_dict = H5FileHandler.load_h5_file(parcel_fname)
        print(f"✅ Loaded parcel signals from: {parcel_fname}")
        return data_dict["parcel_signals"]
    return None

def compute_and_save_parcel_signals(stc, subject=DEFAULT_SUBJECT, parc="Schaefer2018_200Parcels_7Networks_order",
                                    mode="pca_flip", bem_name = "lcmv"):
    print(f"🚀 Extracting parcel signals using {parc} and mode {mode}...")

 # **🔍 Print the expected file name before attempting to load**
    parcel_fname = os.path.join(OUTPUT_DIR, f"{DEFAULT_SUBJECT}-{parc}-{bem_name}-{mode}-parcel_signals.h5")
    print(f"🔎 Checking if parcel signals file exists: {parcel_fname}")

    # Try to load parcel signals from HDF5
    parcel_signals = load_parcel_signals_h5(parc, mode, bem_name)

    if parcel_signals is not None:
        print(f"✅ Parcel signals successfully loaded from: {parcel_fname}")
        print(f"📏 Loaded parcel signals shape: {parcel_signals.shape}")
        return parcel_signals  # ✅ File found and loaded

    # **🛑 If we reach this point, the file was NOT found, so we compute it**
    print(f"⚠️ Parcel signals file NOT found. Computing signals from STC...")
    subject = DEFAULT_SUBJECT
    labels_lh = mne.read_labels_from_annot(subject, parc=parc, hemi="lh", subjects_dir=SUBJECTS_DIR)
    labels_rh = mne.read_labels_from_annot(subject, parc=parc, hemi="rh", subjects_dir=SUBJECTS_DIR)
    labels = labels_lh + labels_rh

    src_fs = mne.read_source_spaces(SRC_ICO5)
    parcel_signals = mne.extract_label_time_course(stc, labels, src=src_fs, mode=mode)
    save_parcel_signals_h5(parcel_signals, bem_name, parc, mode)

    return parcel_signals
