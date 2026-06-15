import numpy as np
import torch
from torch.utils.data import DataLoader

def calculate_pos_weight(train_loader: DataLoader) -> torch.Tensor:
    classes = []

    for _, y in train_loader:
        classes.append(y.numpy())

    classes = np.concatenate(classes, axis=0).reshape(-1).astype(np.int64)
    num_pos = np.sum(classes)
    num_neg = len(classes) - num_pos

    return torch.tensor([num_neg / num_pos], dtype=torch.float32)
