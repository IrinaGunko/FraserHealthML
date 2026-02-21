import logging
from typing import Optional, Union, List, Literal
import mne


class BandpassFilter:
    @staticmethod
    def apply(
        raw: mne.io.Raw,
        l_freq: Optional[float] = 1.0,
        h_freq: Optional[float] = 50.0,
        picks: Union[str, List[str], None] = None,
        filter_length: Union[Literal["auto"], int] = "auto",
        l_trans_bandwidth: Union[Literal["auto"], float] = "auto",
        h_trans_bandwidth: Union[Literal["auto"], float] = "auto",
        n_jobs: Optional[Union[int, Literal["cuda"]]] = None,
        method: Literal["fir", "iir"] = "fir",
        iir_params: Optional[dict] = None,
        phase: Literal["zero", "zero-double", "minimum"] = "zero",
        fir_window: Literal["hamming", "hann", "blackman", "bartlett"] = "hamming",
        fir_design: Literal["firwin", "firwin2"] = "firwin",
        skip_by_annotation: Union[Literal["edge", "bad_acq_skip"], List[str]] = ("edge", "bad_acq_skip"),
        pad: Literal["reflect_limited", "edge", "constant", "mean", "zeros"] = "reflect_limited",
        verbose: Optional[Union[bool, Literal["info", "debug", "warning"]]] = None,
        copy: bool = True,
    ) -> mne.io.Raw:

        logger = logging.getLogger(__name__)
        logger.info("Starting bandpass filtering.")

        # If copy is True, create a copy of the data
        if copy:
            raw = raw.copy()

        params = {
            "l_freq": l_freq,
            "h_freq": h_freq,
            "picks": picks,
            "filter_length": filter_length,
            "l_trans_bandwidth": l_trans_bandwidth,
            "h_trans_bandwidth": h_trans_bandwidth,
            "n_jobs": n_jobs,
            "method": method,
            "iir_params": iir_params,
            "phase": phase,
            "fir_window": fir_window,
            "fir_design": fir_design,
            "skip_by_annotation": skip_by_annotation,
            "pad": pad,
            "verbose": verbose,
        }

        try:
            raw.filter(**params)
            logger.info("Bandpass filtering applied successfully.")
        except Exception as e:
            logger.error(f"Error during bandpass filtering: {e}")
            raise

        return raw