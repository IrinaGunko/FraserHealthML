import re
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mne_label_to_sa_label(label_name):
    if label_name.endswith('-lh'):
        return f"lh_{label_name.replace('-lh', '')}"
    elif label_name.endswith('-rh'):
        return f"rh_{label_name.replace('-rh', '')}"
    else:
        return label_name

LABEL_TO_NETWORK = {
    'VisCent': 'Visual A',
    'VisPeri': 'Visual B',
    'SomMotA': 'Somatomotor A',
    'SomMotB': 'Somatomotor B',
    'DorsAttnA': 'Dorsal Attention A',
    'DorsAttnB': 'Dorsal Attention B',
    'SalVentAttnA': 'Salience/Ventral Attention A',
    'SalVentAttnB': 'Salience/Ventral Attention B',
    'LimbicA': 'Limbic A',
    'LimbicB': 'Limbic B',
    'ContA': 'Control A',
    'ContB': 'Control B',
    'ContC': 'Control C',
    'DefaultA': 'Default A',
    'DefaultB': 'Default B',
    'DefaultC': 'Default C',
    'TempPar': 'Temporal Parietal'
}


LABEL_TO_PARCEL = {
    'AntTemp': 'anterior temporal', 'Aud': 'auditory', 'Cent': 'central',
    'Cinga': 'cingulate anterior', 'Cingm': 'mid-cingulate', 'Cingp': 'cingulate posterior',
    'ExStr': 'extrastriate cortex', 'ExStrInf': 'extra-striate inferior', 'ExStrSup': 'extra-striate superior',
    'FEF': 'frontal eye fields', 'FPole': 'frontal pole', 'FrMed': 'frontal medial',
    'FrOper': 'frontal operculum', 'IFG': 'inferior frontal gyrus', 'Ins': 'insula',
    'IPL': 'inferior parietal lobule', 'IPS': 'intraparietal sulcus', 'OFC': 'orbital frontal cortex',
    'ParMed': 'parietal medial', 'ParOcc': 'parietal occipital', 'ParOper': 'parietal operculum',
    'pCun': 'precuneus', 'pCunPCC': 'precuneus posterior cingulate cortex', 'PFCd': 'dorsal prefrontal cortex',
    'PFCl': 'lateral prefrontal cortex', 'PFCld': 'lateral dorsal prefrontal cortex', 'PFClv': 'lateral ventral prefrontal cortex',
    'PFCm': 'medial prefrontal cortex', 'PFCmp': 'medial posterior prefrontal cortex', 'PFCv': 'ventral prefrontal cortex',
    'PHC': 'parahippocampal cortex', 'PostC': 'post central', 'PrC': 'precentral',
    'PrCd': 'precentral dorsal', 'PrCv': 'precentral ventral', 'RSC': 'retrosplenial cortex',
    'Rsp': 'retrosplenial', 'S2': 'S2', 'SPL': 'superior parietal lobule', 'ST': 'superior temporal',
    'Striate': 'striate cortex', 'StriCal': 'striate calcarine', 'Temp': 'temporal',
    'TempOcc': 'temporal occipital', 'TempPar': 'temporal parietal', 'TempPole': 'temporal pole', 'SomMotA': 'Somatomotor A'
}

def extract_parcel_code(label: str) -> str:
    parts = label.split('_')
    if len(parts) >= 2:
        return parts[-2]
    else:
        logger.warning(f"Could not extract parcel code from label '{label}'")
        return None


def parse_label_name(label: str) -> dict:
    original_label = label
    label = label.replace('-lh', '').replace('-rh', '')  # clean MNE suffix
    parts = label.split('_')

    logger.debug(f"Parsing label: {original_label} -> parts: {parts}")

    if len(parts) < 5:
        logger.warning(f"Label format unexpected: {original_label}")
        return {
            'network_code': 'Unknown',
            'network_name': original_label,
            'parcel_code': 'Unknown',
            'parcel_name': original_label
        }

    network_code = parts[3]
    parcel_code = parts[4]
    logger.debug(f"Extracted network_code: '{network_code}', parcel_code: '{parcel_code}'")
    parcel_name = LABEL_TO_PARCEL.get(parcel_code, parcel_code)
    network_name = LABEL_TO_NETWORK.get(network_code, network_code)

    return {
        'network_code': network_code,
        'network_name': network_name,
        'parcel_code': parcel_code,
        'parcel_name': parcel_name
    }