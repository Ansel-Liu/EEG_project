from pathlib import Path

from torch import nn

from data_utils import get_seizure_datasets, get_patient_datasets, get_window_datasets
from io_data import DataIOBase, NPZDataIO
from models import EEGChannelInputFusion, EEGChannelFeatureFusion
from models.eeg_fusion_cnn import EEG_CNN_LSTM


def get_data_io(typ: str) -> DataIOBase:
    if typ == "npz":
        return NPZDataIO()

    raise ValueError(f"Unknown data io {typ}")


def get_dataset_split(typ: str, data_directory: Path, pat_id_test, val_split: float):
    if typ == "seizure":
        datasets = get_seizure_datasets
    elif typ == "patient":
        datasets = get_patient_datasets
    elif typ == "window":
        datasets = get_window_datasets
    else:
        raise ValueError(f"Unknown dataset type: {typ}")

    return datasets(
        data_directory=data_directory,
        pat_id_test=pat_id_test,
        val_split=val_split,
    )


def get_model(typ: str, eeg_channels: int = 21, out_features: int = 1, dropout: float = 0.2) -> nn.Module:
    if typ == "input_fusion":
        model = EEGChannelInputFusion
    elif typ == "feature_fusion":
        model = EEGChannelFeatureFusion
    elif typ == "cnn_lstm":
        model = EEG_CNN_LSTM
    else:
        raise ValueError(f"Unknown model type {typ}")

    return model(
        eeg_channels=eeg_channels,
        out_features=out_features,
        dropout=dropout,
    )