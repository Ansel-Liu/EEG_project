from pathlib import Path
import numpy as np
import pandas as pd

from .io_raw_data import DataIOBase
from .io_meta import MetaDataIO

class EEGFileLoader:
    """
    Coordinator class to load paired EEG raw data (.npz) and metadata (.parquet).
    """
    def __init__(self, raw_data_loader: DataIOBase, meta_data_loader: MetaDataIO):
        # Dependency Injection: highly decoupled and testable design
        self.raw_data_loader = raw_data_loader
        self.meta_data_loader = meta_data_loader

    def load_eeg_window_data(self, filepath: Path, data_name: str = "EEG_win") -> np.ndarray:
        """Loads the raw EEG window array from a specific file."""
        data = self.raw_data_loader.load(filepath=filepath)
        return data[data_name]

    def load_eeg_window_meta(self, filepath: Path) -> pd.DataFrame:
        """Loads the metadata dataframe from a specific file."""
        return self.meta_data_loader.load(filepath=filepath)

    def load_eeg_window_data_and_meta(
            self,
            eeg_window_data_filepath: Path,
            data_name: str = "EEG_win",
            replace_in_meta_filename: list[str] = None) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Safely loads and pairs the EEG data with its corresponding metadata 
        by resolving the file naming conventions.
        """
        if replace_in_meta_filename is None:
            replace_in_meta_filename = ["EEGwindow", "metadata"]

        # Safely replace string to find the matching metadata file
        eeg_window_meta_filename = eeg_window_data_filepath.stem.replace(
            replace_in_meta_filename[0], replace_in_meta_filename[1]
        )
        
        # Load raw data
        data = self.load_eeg_window_data(filepath=eeg_window_data_filepath, data_name=data_name)
        
        # Construct path and load metadata
        meta_filepath = eeg_window_data_filepath.parent / f"{eeg_window_meta_filename}.parquet"
        meta = self.load_eeg_window_meta(filepath=meta_filepath)

        return data, meta

    def get_extension(self) -> str:
        return self.raw_data_loader.get_extension()