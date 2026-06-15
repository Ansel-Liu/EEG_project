from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def make_subject_groupkfold_splits(metadata: pd.DataFrame, n_splits: int):
    groups = metadata["patient_id"].to_numpy()
    dummy_X = np.zeros(len(metadata))
    y = metadata["class"].to_numpy()

    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(splitter.split(dummy_X, y, groups=groups), start=1):
        folds.append({
            "fold": fold_idx,
            "train_val_idx": np.array(train_val_idx),
            "test_idx": np.array(test_idx),
        })
    return folds


def make_subject_val_split(metadata_subset: pd.DataFrame, val_fraction: float, seed: int):
    groups = metadata_subset["patient_id"].to_numpy()
    dummy_X = np.zeros(len(metadata_subset))

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_local, val_local = next(splitter.split(dummy_X, groups=groups))

    train_idx = metadata_subset.iloc[train_local]["window_index"].to_numpy()
    val_idx = metadata_subset.iloc[val_local]["window_index"].to_numpy()
    return train_idx, val_idx


def describe_split(metadata: pd.DataFrame, name: str, idx: np.ndarray) -> Dict[str, object]:
    part = metadata.iloc[idx]
    summary = {
        "windows": int(len(part)),
        "positives": int((part["class"] == 1).sum()),
        "negatives": int((part["class"] == 0).sum()),
        "patients": int(part["patient_id"].nunique()),
        "recordings": int(part["filename"].nunique()),
        "intervals": int(part["global_interval"].nunique()),
        "patient_ids": sorted(part["patient_id"].unique().tolist()),
    }

    print(f"{name}:")
    print(f"  windows   : {summary['windows']}")
    print(f"  positives : {summary['positives']}")
    print(f"  negatives : {summary['negatives']}")
    print(f"  patients  : {summary['patients']}")
    print(f"  recordings: {summary['recordings']}")
    print(f"  intervals : {summary['intervals']}")
    print(f"  patient_ids: {summary['patient_ids']}")
    print()
    return summary
