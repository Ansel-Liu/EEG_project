import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def compute_channel_stats(X_train: np.ndarray):
    """
    X_train shape: [N, 21, 128]
    returns mean/std shape [21, 1]
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    mean = X_train.mean(axis=(0, 2))[:, None]
    std = X_train.std(axis=(0, 2))[:, None]
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_eeg(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return ((X - mean[None, :, :]) / std[None, :, :]).astype(np.float32)


class EEGWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    y_int = np.asarray(y).astype(int)
    class_counts = np.bincount(y_int)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_int]
    sample_weights = torch.DoubleTensor(sample_weights)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
