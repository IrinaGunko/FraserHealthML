import os
import h5py
import numpy as np
import datetime

class H5FileHandler:

    @staticmethod
    def load_hdf5_parcel_signals(hdf5_path):
        with h5py.File(hdf5_path, 'r') as f:
            parcel_signals = f['parcel_signals'][:]
            label_names = [name.decode() for name in f['label_names'][:]]
            metadata = dict(f.attrs)
        return parcel_signals, label_names, metadata

    @staticmethod
    def save_h5_file(filepath, data_dict, attr_dict=None, overwrite=False):
        if os.path.exists(filepath) and not overwrite:
            print(f"File already exists: {filepath}. Skipping save.")
            return
        with h5py.File(filepath, "w") as f:
            for key, value in data_dict.items():
                f.create_dataset(key, data=value, compression="gzip", compression_opts=9)
            if attr_dict:
                for key, value in attr_dict.items():
                    f.attrs[key] = value

        print(f"Data saved successfully to: {filepath}")

    @staticmethod
    def load_hdf5_parcel_data(h5_path):
        with h5py.File(h5_path, "r") as f:
            signals = f["parcel_signals"][:]
            labels = [l.decode() for l in f["label_names"][:]]
            metadata = {k: v.decode() if isinstance(v, bytes) else v for k, v in f["metadata"].attrs.items()}
        return signals, labels, metadata

    @staticmethod
    def load_h5_file(filepath):
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None
        data_dict = {}
        attr_dict = {}
        with h5py.File(filepath, "r") as f:
            for key in f.keys():
                data_dict[key] = f[key][:]
            for key in f.attrs.keys():
                attr_dict[key] = f.attrs[key]
        print(f"Loaded data from {filepath}")
        return data_dict, attr_dict

    @staticmethod
    def save_parcel_signals_to_hdf5(parcel_signals, labels, metadata, output_path):
        with h5py.File(output_path, 'w') as f:
            f.attrs['created_on'] = str(datetime.datetime.now())
            f.attrs.update(metadata)

            f.create_dataset('parcel_signals', data=parcel_signals)
            label_names = [lbl.name for lbl in labels]
            f.create_dataset('label_names', data=np.array(label_names, dtype='S'))