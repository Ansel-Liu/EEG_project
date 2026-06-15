import os
import re
import glob
import numpy as np
import pandas as pd


REQUIRED_METADATA_COLS = ["class", "filename_interval", "global_interval", "filename"]


def load_npz_array(npz_path: str) -> np.ndarray:
    """
    Load one patient's EEG windows from an .npz file and force the result to be
    a real numeric array of shape [N, 21, 128] with dtype float32.
    """
    npz_data = np.load(npz_path, allow_pickle=True)

    if len(npz_data.files) == 1:
        key = npz_data.files[0]
        X = npz_data[key]
    else:
        preferred_keys = ["EEG_win", "arr_0", "x", "X", "windows"]
        found = None
        for key in preferred_keys:
            if key in npz_data.files:
                found = key
                break

        if found is None:
            raise ValueError(
                f"Could not determine EEG array key in {npz_path}. "
                f"Available keys: {npz_data.files}"
            )
        X = npz_data[found]

    # Unwrap common object containers.
    if isinstance(X, np.ndarray) and X.dtype == object:
        if X.shape == ():
            X = X.item()
        elif len(X) == 1:
            X = X[0]
        else:
            # Convert element by element in case this is an object array of windows.
            X = np.stack([np.asarray(x, dtype=np.float32) for x in X], axis=0)

    # If the loaded object is dict-like, try common keys.
    if isinstance(X, dict):
        for key in ["EEG_win", "arr_0", "x", "X", "windows"]:
            if key in X:
                X = X[key]
                break

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape [N,21,128], got {X.shape}")

    return X


def extract_patient_id_from_name(name: str) -> str:
    """
    Extract patient ID such as chb01 from filenames like:
    chb01_seizure_EEGwindow_1.npz
    chb01_something.parquet
    """
    base = os.path.basename(name)
    match = re.search(r"(chb\d+)", base.lower())
    if match:
        return match.group(1)
    return base.split("_")[0].lower()


def find_patient_file_pairs(data_dir: str):
    """
    Find .npz and .parquet files and pair them by patient ID.
    Assumes one NPZ and one parquet per patient.
    """
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")

    npz_map = {}
    for path in npz_files:
        pid = extract_patient_id_from_name(path)
        npz_map.setdefault(pid, []).append(path)

    parquet_map = {}
    for path in parquet_files:
        pid = extract_patient_id_from_name(path)
        parquet_map.setdefault(pid, []).append(path)

    common_patients = sorted(set(npz_map.keys()) & set(parquet_map.keys()))
    if not common_patients:
        raise ValueError("No matching patient IDs found between .npz and .parquet files.")

    pairs = []
    for pid in common_patients:
        if len(npz_map[pid]) != 1:
            raise ValueError(
                f"Expected exactly 1 npz for {pid}, found {len(npz_map[pid])}: {npz_map[pid]}"
            )
        if len(parquet_map[pid]) != 1:
            raise ValueError(
                f"Expected exactly 1 parquet for {pid}, found {len(parquet_map[pid])}: {parquet_map[pid]}"
            )

        pairs.append(
            {
                "patient_id": pid,
                "npz_path": npz_map[pid][0],
                "metadata_path": parquet_map[pid][0],
            }
        )

    return pairs


def load_single_patient_pair(npz_path: str, metadata_path: str, patient_id: str):
    X = load_npz_array(npz_path)
    metadata = pd.read_parquet(metadata_path)

    missing = [c for c in REQUIRED_METADATA_COLS if c not in metadata.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns in {metadata_path}: {missing}")

    if len(X) != len(metadata):
        raise ValueError(
            f"Mismatch for patient {patient_id}: "
            f"{len(X)} windows in {npz_path} vs {len(metadata)} rows in {metadata_path}"
        )

    metadata = metadata.copy()
    metadata["patient_id"] = patient_id
    metadata["source_npz"] = os.path.basename(npz_path)
    metadata["source_parquet"] = os.path.basename(metadata_path)

    return X, metadata


def load_all_patients(data_dir: str):
    pairs = find_patient_file_pairs(data_dir)

    X_parts = []
    meta_parts = []
    running_offset = 0

    print(f"Found {len(pairs)} patient file pairs.\n")

    for pair in pairs:
        patient_id = pair["patient_id"]
        npz_path = pair["npz_path"]
        metadata_path = pair["metadata_path"]

        X_i, meta_i = load_single_patient_pair(npz_path, metadata_path, patient_id)

        local_n = len(meta_i)
        meta_i = meta_i.copy()
        meta_i["local_window_index"] = np.arange(local_n)
        meta_i["window_index"] = np.arange(running_offset, running_offset + local_n)

        X_parts.append(np.asarray(X_i, dtype=np.float32))
        meta_parts.append(meta_i)

        print(
            f"{patient_id}: "
            f"X={X_i.shape}, "
            f"metadata={meta_i.shape}, "
            f"positives={(meta_i['class'] == 1).sum()}, "
            f"negatives={(meta_i['class'] == 0).sum()}"
        )

        running_offset += local_n

    X = np.concatenate([np.asarray(x, dtype=np.float32) for x in X_parts], axis=0)
    metadata = pd.concat(meta_parts, axis=0, ignore_index=True)

    if len(X) != len(metadata):
        raise ValueError(f"Final mismatch after concatenation: {len(X)} vs {len(metadata)}")

    return X, metadata, pairs


def inspect_data(X: np.ndarray, metadata: pd.DataFrame) -> None:
    print("\n===== DATA INSPECTION =====")
    print("X shape:", X.shape)
    print("X dtype:", X.dtype)
    print("Metadata shape:", metadata.shape)
    print("Label distribution:")
    print(metadata["class"].value_counts(dropna=False).sort_index())
    print("Unique patients:", metadata["patient_id"].nunique())
    print("Unique recordings:", metadata["filename"].nunique())
    print("Unique global intervals:", metadata["global_interval"].nunique())
    print("===========================\n")
