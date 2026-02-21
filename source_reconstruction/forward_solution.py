import mne
import os
from utils.load_eeg import load_eeg_data_single_file
from config import (
    SUBJECTS_DIR, TRANS, SRC_ICO5,  BEM, OUTPUT_DIR, DEFAULT_SUBJECT
)

def read_forward_solution(src_type="cortical", output_dir=OUTPUT_DIR):
    fwd_fname = os.path.join(output_dir, f"{DEFAULT_SUBJECT}-{src_type}-fwd.fif")
    if os.path.exists(fwd_fname):
        print(f"✅ Loading existing forward solution from: {fwd_fname}")
        return mne.read_forward_solution(fwd_fname, verbose=True)
    return None

def save_forward_solution(fwd, src_type="cortical", output_dir=OUTPUT_DIR):
    fwd_fname = os.path.join(output_dir, f"{DEFAULT_SUBJECT}-{src_type}-fwd.fif")
    print(f"💾 Saving forward solution to: {fwd_fname}")
    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)

def make_forward_solution(raw):
    trans_fs = mne.read_trans(TRANS)
    src_fs = mne.read_source_spaces(SRC_ICO5)
    bem_fs = mne.read_bem_solution(BEM)
    fwd = mne.make_forward_solution(
        raw.info, trans=trans_fs, src=src_fs, bem=bem_fs, eeg=True, ignore_ref = True, mindist=5.0, verbose=True
    )
    return fwd
