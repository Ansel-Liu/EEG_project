import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, fields


@dataclass
class DataLabel:
    X: np.ndarray
    y: np.ndarray
    file_interval: np.ndarray
    global_interval: np.ndarray
    patient_id: np.ndarray
    filename_index: np.ndarray


def label_data(data: np.ndarray, metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    label = np.array(metadata["class"].tolist()).reshape((-1, 1))
    file_interval = np.array(metadata["filename_interval"].tolist())
    global_interval = np.array(metadata["global_interval"].tolist())
    pat_info = np.array(metadata["filename"].str.split(r"[_.]").tolist())

    return asdict(
        obj=DataLabel(
            X=data,
            y=label,
            file_interval=file_interval,
            global_interval=global_interval,
            patient_id=pat_info[:, 0],
            filename_index=pat_info[:, 1],
        )
    )


def get_empty_data_dict() -> dict[str, list]:
        return {k.name: [] for k in fields(DataLabel)}
