import mne
from pathlib import Path
from config import EEG_FOLDER

def load_eeg_data_single_file(eeg_file_path, montage_name="standard_1020"):

    raw = mne.io.read_raw_edf(eeg_file_path, preload=True)
    print(raw.ch_names)
    if "Status" in raw.ch_names:
        raw.drop_channels(["Status"])
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage)
    mne.datasets.eegbci.standardize(raw)
    print(f"✅ EEG data loaded successfully from {Path(eeg_file_path).name} with montage {montage_name}")
    return raw