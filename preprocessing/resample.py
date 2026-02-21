import logging
from typing import Optional, Union
import mne


class ResampleData:
    @staticmethod
    def apply(
        raw: mne.io.Raw,
        sfreq: float,
        npad: Union[int, str] = "auto",
        window: Union[str, tuple] = "auto",
        stim_picks: Optional[Union[list, None]] = None,
        n_jobs: Optional[Union[int, str]] = None,
        pad: str = "auto",
        method: str = "fft",
        verbose: Optional[Union[bool, str, int]] = None,
        copy: bool = True,
    ) -> mne.io.Raw:

        logger = logging.getLogger(__name__)
        logger.info("Starting resampling.")

        # Ensure a copy is created if requested
        if copy:
            raw = raw.copy()

        # Log user-set and default parameters
        params = {
            "sfreq": sfreq,
            "npad": npad,
            "window": window,
            "stim_picks": stim_picks,
            "n_jobs": n_jobs,
            "pad": pad,
            "method": method,
            "verbose": verbose,
        }
        logger.info("Parameters used for resampling:")
        for key, value in params.items():
            logger.info(f"{key}: {value}")

        try:
            raw.resample(
                sfreq=sfreq,
                npad=npad,
                window=window,
                stim_picks=stim_picks,
                n_jobs=n_jobs,
                pad=pad,
                method=method,
                verbose=verbose,
            )
            logger.info(f"Resampling to {sfreq} Hz completed successfully.")
        except Exception as e:
            logger.error(f"Error during resampling: {e}")
            raise

        return raw