# eeg_features/features/diptest.py

import numpy as np
import diptest

def hartigans_diptest(values: np.ndarray) -> dict:
    clean_values = values[~np.isnan(values)]

    if len(clean_values) < 5:
        return {"dip_stat": np.nan, "p_value": np.nan}

    try:
        dip, pval = diptest.diptest(clean_values)
    except Exception:
        dip, pval = np.nan, np.nan

    return {"dip_stat": dip, "p_value": pval}
