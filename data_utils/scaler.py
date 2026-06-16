import abc
import numpy as np
from sklearn.preprocessing import StandardScaler

class ScalerBase(abc.ABC):
    def __init__(self):
        self.scaler = self._get_scaler()

    @abc.abstractmethod
    def _get_scaler(self):
        pass

    def scale_X(self, X: np.ndarray) -> np.ndarray:
        """
        Scales the input tensor X of shape (N, C, T) where:
        N = Samples, C = Channels, T = Time steps.
        It flattens N and C to scale each time step independently.
        """
        original_shape = X.shape

        # Reshape data to 2D: (N * C, T) to align with Equation (1) & (2) in report
        X = np.reshape(X, (np.prod(original_shape[:-1]), original_shape[-1]))
        
        # Apply transformation
        X = self._transform(X=X)
        
        # Reshape back to original 3D form: (N, C, T)
        X = np.reshape(X, original_shape)

        return X

    @abc.abstractmethod
    def _transform(self, X: np.ndarray) -> np.ndarray:
        pass

class StandardScalerTransposed(ScalerBase):
    """
    Standardizes each time step independently to zero mean and unit variance.
    Implementation corresponds to the Z-score normalization in the project report.
    """
    def __init__(self):
        super().__init__()

    def _get_scaler(self):
        return StandardScaler()

    def _transform(self, X: np.ndarray) -> np.ndarray:
        # Transpose to fit StandardScaler (which scales columns independently)
        return self.scaler.fit_transform(X.T).T

def get_scaler(scaler_name: str) -> ScalerBase:
    if scaler_name == "standardTransposed":
        return StandardScalerTransposed()
    raise ValueError(f"Unknown scaler: {scaler_name}")