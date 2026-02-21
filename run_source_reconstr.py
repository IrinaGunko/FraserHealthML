import mne
import os
import argparse
import glob
from utils.HDF5Handler import H5FileHandler
from config import PreProcessingConfig
from config import PyPrepConfig
from preprocessing.filtering import preprocess_raw
from preprocessing.pyprep_wrapper import apply_pyprep
from utils.load_eeg import load_eeg_data_single_file
from source_reconstruction.beamformer_amoiseev import compute_beamformer_stc
from source_reconstruction.forward_solution import make_forward_solution
from parcellation.get_parcellation import get_parcellation
from utils.extract_metadata import extract_metadata_from_filename
from config import OUTPUT_DIR, SRC_ICO5

def process_file(filepath, parc_name):
    prepropConfig = PreProcessingConfig()
    pyprepConfig = PyPrepConfig()
    raw = load_eeg_data_single_file(filepath)
    raw = preprocess_raw(raw, prepropConfig)
    raw = apply_pyprep(raw, pyprepConfig)
    raw.set_eeg_reference("average", projection=True)
    fwd = make_forward_solution(raw)
    stc = compute_beamformer_stc(raw, fwd, return_stc=True)[0]
    parcellation = get_parcellation(parc_name)
    labels = parcellation.get_labels()
    src = mne.read_source_spaces(SRC_ICO5)
    parcel_signals = mne.extract_label_time_course(stc, labels, src=src, mode="pca_flip")
    subject_id, session, task, acquisition = extract_metadata_from_filename(filepath)
    metadata = {
        "subject_id": subject_id,
        "session": session,
        "task": task,
        "acquisition": acquisition,
        "sampling_rate": raw.info['sfreq'],
        "parcellation": parc_name
    }

    output_path = os.path.join(OUTPUT_DIR, f"{subject_id}_{session}_{task}_{acquisition}_{parc_name}.h5")
    H5FileHandler.save_parcel_signals_to_hdf5(parcel_signals, labels, metadata, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EEG preprocessing and source analysis")
    parser.add_argument("--filepath", help="Path to a specific EEG .edf file")
    parser.add_argument("--index", type=int, help="Index of the EEG file to process (for SLURM job arrays)")
    parser.add_argument("--data-folder", default="raw_eeg", help="Folder with .edf EEG files")
    parser.add_argument("--parc", default="schaefer", help="Parcellation name (e.g., schaefer, hcpmmp)")

    args = parser.parse_args()

    if args.filepath:
        process_file(args.filepath, args.parc)
    elif args.index is not None:
        edf_files = sorted(glob.glob(os.path.join(args.data_folder, "*.edf")))
        if args.index >= len(edf_files):
            raise IndexError(f"Index {args.index} out of range for {len(edf_files)} files in {args.data_folder}")
        filepath = edf_files[args.index]
        process_file(filepath, args.parc)
    else:
        raise ValueError("Either --filepath or --index must be provided")