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
        original_shape = X.shape

        # reshape the data to a 2D shape
        X = np.reshape(X, (np.prod(original_shape[:-1]), original_shape[-1]))
        # scale the data
        X = self._transform(X=X)
        # reshape the data back in original form
        X = np.reshape(X, original_shape)

        return X

    @abc.abstractmethod
    def _transform(self, X) -> np.ndarray:
        pass


class StandardScalerTransposed(ScalerBase):
    """
    This standard scaler scales the data by frequencies.
    """
    def __init__(self):
        super().__init__()

    def _get_scaler(self):
        return StandardScaler()

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(X.T).T


def get_scaler(scaler_name: str) -> ScalerBase:
    if scaler_name == "standardTransposed":
        return StandardScalerTransposed()

    raise ValueError(f"Unknown scaler {scaler_name}")