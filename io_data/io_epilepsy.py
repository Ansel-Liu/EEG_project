from pathlib import Path

import numpy as np
import pandas as pd

from .io_raw_data import DataIOBase
from .io_meta import MetaDataIO


class DataLoader:
    def __init__(self, raw_data_loader: DataIOBase, meta_data_loader: MetaDataIO):
        self.raw_data_loader = raw_data_loader
        self.meta_data_loader = meta_data_loader

    def load_eeg_window_data(self, filepath: Path, data_name: str = "EEG_win") -> np.ndarray:
        """
        Load the EEG window data from a npz file.

        :param filepath: Path to the EEG window npz file.
        :param data_name: EEG window name in the npz file.
        :return: The EEG window data.
        """
        data = self.raw_data_loader.load(filepath=filepath)

        return data[data_name]


    def load_eeg_window_meta(self, filepath: Path) -> pd.DataFrame:
        """
        Load the metadata from a parquet file.

        :param filepath: Path to the parquet meta file.
        :return: The metadata.
        """
        return self.meta_data_loader.load(filepath=filepath)


    def load_eeg_window_data_and_meta(
            self,
            eeg_window_data_filepath: Path,
            data_name: str = "EEG_win",
            replace_in_meta_filename=None) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Load the data and metadata from a EEG window file.
        By default, 'EEGwindow' will be replaced with 'metadata' in the meta filepath.

        :param eeg_window_data_filepath: Path to the EEG window npz file.
        :param data_name: EEG window name in the npz file.
        :param replace_in_meta_filename: A list with the strings to replace for the metadata file.
        :return: EEG window data and metadata.
        """
        if replace_in_meta_filename is None:
            replace_in_meta_filename = ["EEGwindow", "metadata"]

        eeg_window_meta_filename = eeg_window_data_filepath.stem.replace(*replace_in_meta_filename)
        data = self.load_eeg_window_data(filepath=eeg_window_data_filepath, data_name=data_name)
        meta = self.load_eeg_window_meta(filepath=eeg_window_data_filepath.parent / f"{eeg_window_meta_filename}.parquet")

        return data, meta

    def get_extension(self):
        return self.raw_data_loader.get_extension()