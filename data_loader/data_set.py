import numpy as np
import torch
from torch.utils.data import Dataset

class NpzDataset(Dataset):
    """
    A custom PyTorch Dataset for loading in-memory NumPy arrays.
    
    This dataset wraps the preprocessed EEG features (X) and their 
    corresponding binary labels (y), converting them into PyTorch tensors 
    ready for model training, validation, and testing.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Args:
            X (np.ndarray): The input feature matrix (e.g., EEG signals).
            y (np.ndarray): The corresponding labels (0 for normal, 1 for seizure).
        """
        # Convert NumPy arrays to PyTorch tensors (float32 is standard for neural networks)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the feature and label tensor at the specified index.
        
        Args:
            idx (int): The index of the sample.
            
        Returns:
            tuple: (feature_tensor, label_tensor)
        """
        return self.X[idx], self.y[idx]