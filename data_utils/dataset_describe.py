import numpy as np


def describe_split(split: dict[str, np.ndarray], name: str) -> dict[str, object]:
    summary = {
        "windows": int(split["y"].shape[0]),
        "seizure": int(np.sum(split["y"][:, 0] == 1)),
        "no_seizure": int(np.sum(split["y"][:, 0] == 0)),
        "patients": int(np.unique(split["patient_id"]).shape[0]),
        "recordings": int(np.unique(split["filename_index"]).shape[0]),
        "intervals": int(np.unique(split["global_interval"]).shape[0]),
        "patient_ids": sorted(np.unique(split["patient_id"]).tolist()),
    }

    print(f"{name}:")
    print(f"  windows   : {summary['windows']}")
    print(f"  seizure : {summary['seizure']}")
    print(f"  no seizure : {summary['no_seizure']}")
    print(f"  patients  : {summary['patients']}")
    print(f"  recordings: {summary['recordings']}")
    print(f"  intervals : {summary['intervals']}")
    print(f"  patient_ids: {summary['patient_ids']}")
    print()

    return summary