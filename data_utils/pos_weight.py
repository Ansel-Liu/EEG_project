import numpy as np
import torch
from torch.utils.data import DataLoader

def calculate_pos_weight(train_loader: DataLoader) -> torch.Tensor:
    """
    Dynamically calculates the positive weight for imbalanced datasets.
    
    This is used in BCEWithLogitsLoss to scale the loss of positive (seizure) 
    samples, preventing the model from collapsing into majority class prediction.
    
    Formula: pos_weight = Number of Negative Samples / Number of Positive Samples
    """
    classes = []

    # Iterate through the loader to collect all ground truth labels
    for _, y in train_loader:
        classes.append(y.numpy())

    classes = np.concatenate(classes, axis=0).reshape(-1).astype(np.int64)
    
    num_pos = np.sum(classes)
    num_neg = len(classes) - num_pos

    # Prevent division by zero if no positive samples exist
    if num_pos == 0:
        return torch.tensor([1.0], dtype=torch.float32)

    return torch.tensor([num_neg / num_pos], dtype=torch.float32)