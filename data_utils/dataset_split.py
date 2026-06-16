from pathlib import Path
import numpy as np

from .utils import get_paths
from .data_label import get_empty_data_dict

def get_np_train_val_and_test(train_val_dataset, test_dataset, val_split: float = 0.2):
    """Utility to finalize the splits into Training, Validation, and Test dictionaries."""
    for key in train_val_dataset.keys():
        train_val_dataset[key] = np.concatenate(train_val_dataset[key], axis=0)
        test_dataset[key] = np.concatenate(test_dataset[key], axis=0)

    pat_ids = train_val_dataset["patient_id"]
    unique_patient_ids = np.unique(train_val_dataset["patient_id"])

    train_mask = np.zeros(train_val_dataset["y"].shape[0], dtype=bool)
    split_idx = int(len(unique_patient_ids) * (1. - val_split))
    
    for pat_id in unique_patient_ids[:split_idx]:
        train_mask[pat_ids == pat_id] = True

    train_dataset = get_empty_data_dict()
    val_dataset = get_empty_data_dict()

    for key in train_val_dataset.keys():
        train_dataset[key] = train_val_dataset[key][train_mask]
        val_dataset[key] = train_val_dataset[key][~train_mask]

    return train_dataset, val_dataset, test_dataset

def get_seizure_datasets(data_directory: Path, pat_id_test: str, val_split: float = 0.2):
    """
    Generates datasets using a Seizure-level split.
    CRITICAL: It applies a mask to drop temporally adjacent windows to prevent Data Leakage.
    """
    if not isinstance(pat_id_test, str):
        raise TypeError("Patient ID must be a string for the DataLoader by seizure")

    paths = get_paths(data_directory=data_directory)
    train_val_dataset = get_empty_data_dict()
    test_dataset = get_empty_data_dict()

    for path in paths:
        data = np.load(file=path, allow_pickle=True)
        patient_ids = data["patient_id"]
        unique_patient_ids = np.unique(patient_ids)

        train_mask = np.ones(patient_ids.shape[0], dtype=bool)
        
        if pat_id_test in unique_patient_ids:
            y = data["y"].reshape(-1)
            global_interval = data["global_interval"]
            last_no_seizure_idx = global_interval[y == 0].max()
            last_seizure_idx = global_interval[y == 1].max()
            
            # Prevent temporal data leakage by excluding edge intervals
            train_mask[(global_interval == last_no_seizure_idx) & (y == 0)] = 0
            train_mask[(global_interval == last_seizure_idx) & (y == 1)] = 0

        for key, value in data.items():
            train_val_dataset[key].append(value[train_mask])
            test_dataset[key].append(value[~train_mask])

    return get_np_train_val_and_test(
        train_val_dataset=train_val_dataset,
        test_dataset=test_dataset,
        val_split=val_split  # Fixed typo: val_slit -> val_split
    )


def get_patient_datasets(data_directory: Path, pat_id_test, val_split: float = 0.2):
    if not isinstance(pat_id_test, str):
        raise TypeError("Patient ID must be a string for the DataLoader by patient")

    paths = get_paths(data_directory=data_directory)

    train_val_dataset = get_empty_data_dict()
    test_dataset = get_empty_data_dict()

    for path in paths:
        data = np.load(
            file=path,
            allow_pickle=True,
        )

        patient_ids = data["patient_id"]
        unique_patient_ids = np.unique(patient_ids)
        dataset_ = "train_val"

        if pat_id_test in unique_patient_ids:
            dataset_ = "test"

        for key, value in data.items():
            if dataset_ in "train_val":
                train_val_dataset[key].append(value)
            else:
                test_dataset[key].append(value)

    return get_np_train_val_and_test(
        train_val_dataset=train_val_dataset,
        test_dataset=test_dataset,
        val_slit=val_split
    )


def get_window_datasets(data_directory: Path, pat_id_test, val_split: float = 0.2):
    if not isinstance(pat_id_test, tuple):
        raise TypeError("Patient ID must be a tuple for the DataLoader by patient")
    paths = get_paths(data_directory=data_directory)

    train_val_dataset = get_empty_data_dict()
    test_dataset = get_empty_data_dict()

    for path in paths:
        data = np.load(
            file=path,
            allow_pickle=True,
        )

        patient_ids = data["patient_id"]
        unique_patient_ids = np.unique(patient_ids)

        is_test = []
        for unique_patient_id in unique_patient_ids:
            if unique_patient_id in pat_id_test:
                is_test.append(True)

        g = data["global_interval"]
        y = data["y"]
        train_val_mask = np.ones(y.shape[0], dtype=bool)
        test_mask = np.zeros(y.shape[0], dtype=bool)

        if len(is_test) > 0:
            for l in [0, 1]:
                ones_idx = np.where(y == l)[0]
                g_at_ones = g[ones_idx]
                unique_y = np.unique(g_at_ones)
                mask = g_at_ones[:, None] == unique_y[None, :]
                last_pos_in_ones = mask.cumsum(axis=0).argmax(axis=0)

                last_indices = ones_idx[last_pos_in_ones]
                test_mask[last_indices] = True
                train_val_mask[last_indices] = False

                if l == 1:
                    offsets = np.arange(1, 5)
                    prev_indices =  last_pos_in_ones[:, None] - offsets[None, :]
                    prev_indices = np.clip(prev_indices, 0, len(ones_idx) - 1)

                    prev_four = ones_idx[prev_indices]
                    prev_four_unique = np.unique(prev_four)
                    train_val_mask[prev_four_unique] = False

        for key, value in data.items():
            train_val_dataset[key].append(value[train_val_mask])
            test_dataset[key].append(value[test_mask])

    return get_np_train_val_and_test(
        train_val_dataset=train_val_dataset,
        test_dataset=test_dataset,
        val_slit=val_split
    )