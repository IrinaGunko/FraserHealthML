from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict

@dataclass
class PreProcessingConfig:
    target_freq: int = 256
    lfreq: float = 0.5
    hfreq: float = 55.0
    line_noise: float = 50.0
@dataclass
class PyPrepConfig:
    max_iterations: int = 4
    ransac: bool = True
    channel_wise: bool = False
    random_state: int = 12345
    matlab_strict: bool = False
    filter_method: str = "fir"
    line_noise: float = 50.0

@dataclass
class AlphaAnalysisConfig:
    sfreq: int = 256
    psd_fmin: float = 1
    psd_fmax: float = 49.0
    psd_n_fft: int = 512
    peak_range: tuple = (7.0, 14.0)
    fit_freq_range: tuple = (1, 49.0)
    peak_width_limits: tuple = (1, 12)
    max_n_peaks: int = 5
    min_peak_height: float = 0.1
    peak_threshold: float = 1.0
    aperiodic_mode: str = "fixed"
    kmeans_range_freq: tuple = (2, 6)
    kmeans_range_power: tuple = (2, 5)
    silhouette_min_score: float = 0.2
    random_state: int = 12345
    alpha_start: int = 7
    alpha_end: int = 14
    bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "alpha": (7, 13),
        "beta":  (13, 30),
        "gamma": (30, 55),
    })

import socket
import os
from pathlib import Path

# ========== Detect Environment ==========
HOSTNAME = socket.gethostname()
ON_CLUSTER = any(key in HOSTNAME for key in ["cedar", "graham", "narval", "beluga"])

# ========== Paths ==========
if ON_CLUSTER:
    BASE_DIR     = Path("/scratch/irynag/EEGAtlasMapper")
else:
    BASE_DIR     = Path(__file__).parent

EEG_FOLDER   = str(BASE_DIR / "raw_eeg")
OUTPUT_DIR   = str(BASE_DIR / "eeg_output")
SUBJECTS_DIR = str(BASE_DIR)
H5_FOLDER    = str(BASE_DIR / "h_5")

DEFAULT_SUBJECT = "fsaverage"
PROJ_DIR  = SUBJECTS_DIR
TRANS     = str(BASE_DIR / "fsaverage" / "bem" / "fsaverage-trans.fif")
SRC_ICO5  = str(BASE_DIR / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif")
BEM       = str(BASE_DIR / "fsaverage" / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif")
MNI152    = str(BASE_DIR / "mni152")
S_A_AXIS_RANKS = str(BASE_DIR / "S-A_Axis" / "Sensorimotor_Association_Axis_AverageRanks.csv")