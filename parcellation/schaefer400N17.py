from parcellation.base_parcellation import ParcellationStrategy
import mne
from config import DEFAULT_SUBJECT, SUBJECTS_DIR

class SchaeferParcellation(ParcellationStrategy):
    def get_labels(self):
        parc = "Schaefer2018_400Parcels_17Networks_order"
        lh = mne.read_labels_from_annot(DEFAULT_SUBJECT, parc, hemi="lh", subjects_dir=SUBJECTS_DIR)
        rh = mne.read_labels_from_annot(DEFAULT_SUBJECT, parc, hemi="rh", subjects_dir=SUBJECTS_DIR)
        return lh + rh