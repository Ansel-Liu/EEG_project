import random
from pathlib import Path
from itertools import combinations

from .utils import get_paths


def get_kfold_split(data_directory: Path, n_splits: int = 5, random_seed: int = 42, multiply: bool = False) -> list[str]:
    paths = get_paths(data_directory=data_directory)

    patient_names = [path.name.split("_")[0] for path in paths]

    random.seed(random_seed)
    if multiply:
       patient_names = list(combinations(patient_names, 5))

    kfolds = random.sample(
        population=patient_names,
        k=n_splits
    )

    return kfolds