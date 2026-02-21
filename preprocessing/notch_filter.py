import logging
from typing import Optional, Union, List, Literal
import mne


class NotchFilter:
    @staticmethod
    def apply(
        raw: mne.io.Raw,
        freqs: Union[float, List[float], None],
        picks: Union[str, List[str], slice, None] = None,
        filter_length: Union[str, int] = "auto",
        notch_widths: Optional[Union[float, List[float]]] = None,
        trans_bandwidth: float = 1.0,
        n_jobs: Optional[Union[int, Literal["cuda"]]] = None,
        method: Literal["fir", "iir"] = "fir",
        iir_params: Optional[dict] = None,
        mt_bandwidth: Optional[float] = None,
        p_value: float = 0.05,
        phase: Literal["zero", "minimum", "zero-double", "minimum-half"] = "zero",
        fir_window: Literal["hamming", "hann", "blackman"] = "hamming",
        fir_design: Literal["firwin", "firwin2"] = "firwin",
        pad: str = "reflect_limited",
        skip_by_annotation: Union[str, List[str]] = ("edge", "bad_acq_skip"),
        verbose: Optional[Union[bool, str, int]] = None,
        copy: bool = True,
    ) -> mne.io.Raw:

        logger = logging.getLogger(__name__)
        logger.info("Starting notch filtering.")

        # If copy is True, create a copy of the data
        if copy:
            raw = raw.copy()

        # Parameters and defaults
        params = {
            "freqs": freqs,
            "picks": picks,
            "filter_length": filter_length,
            "notch_widths": notch_widths,
            "trans_bandwidth": trans_bandwidth,
            "n_jobs": n_jobs,
            "method": method,
            "iir_params": iir_params,
            "mt_bandwidth": mt_bandwidth,
            "p_value": p_value,
            "phase": phase,
            "fir_window": fir_window,
            "fir_design": fir_design,
            "pad": pad,
            "skip_by_annotation": skip_by_annotation,
            "verbose": verbose,
        }
        defaults = {
            "filter_length": "auto",
            "notch_widths": None,
            "trans_bandwidth": 1.0,
            "method": "fir",
            "iir_params": None,
            "mt_bandwidth": None,
            "p_value": 0.05,
            "phase": "zero",
            "fir_window": "hamming",
            "fir_design": "firwin",
            "pad": "reflect_limited",
            "skip_by_annotation": ("edge", "bad_acq_skip"),
            "verbose": None,
        }

        # Log parameters
        user_set_params = {
            key: value for key, value in params.items() if value != defaults.get(key)
        }
        default_params = {
            key: value for key, value in params.items() if key not in user_set_params
        }

        logger.info("User-set parameters:")
        for key, value in user_set_params.items():
            logger.info(f"{key}: {value}")

        logger.info("Default parameters:")
        for key, value in default_params.items():
            logger.info(f"{key}: {value}")

        # Apply the notch filter
        try:
            raw.notch_filter(**params)
            logger.info("Notch filtering applied successfully.")
        except Exception as e:
            logger.error(f"Error during notch filtering: {e}")
            raise

        return raw