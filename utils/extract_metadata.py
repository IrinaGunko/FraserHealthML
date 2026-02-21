import re
from typing import Tuple, Optional

def extract_metadata_from_filename(filename: str) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    pattern = r"sub-(\d+)_ses-(\d+)_task-(EyesOpen|EyesClosed)_acq-(pre|post)"
    match = re.search(pattern, filename)

    if not match:
        return None, None, None, None

    subject_id, session, task, acquisition = match.groups()
    return subject_id, int(session), task, acquisition