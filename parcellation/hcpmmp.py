from parcellation.base_parcellation import ParcellationStrategy
import mne
from config import DEFAULT_SUBJECT, SUBJECTS_DIR

class HCPMMPParcellation(ParcellationStrategy):
    def get_labels(self):
        parc = "HCPMMP1"
        lh = mne.read_labels_from_annot(DEFAULT_SUBJECT, parc, hemi="lh", subjects_dir=SUBJECTS_DIR)
        rh = mne.read_labels_from_annot(DEFAULT_SUBJECT, parc, hemi="rh", subjects_dir=SUBJECTS_DIR)
        return lh + rh
