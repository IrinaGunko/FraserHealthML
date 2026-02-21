
from abc import ABC, abstractmethod
import mne

class ParcellationStrategy(ABC):
    @abstractmethod
    def get_labels(self):
        pass
