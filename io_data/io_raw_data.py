import abc
from pathlib import Path

import numpy as np


class DataIOBase(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def load(self, filepath: Path) -> dict[str, np.ndarray]:
        """
        Read the raw data.

        :param filepath: Path to the file.
        :return: Dictionary with data as numpy array.
        """
        pass

    def write(self, filepath: Path, data: dict[str, np.ndarray]) -> None:
        """
        Write data to a file.

        :param filepath: Path to the file.
        :param data: Dictionary of data with name.
        :return: None
        """
        if not filepath.parent.exists():
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if filepath.suffix != self.get_extension():
            filepath += self.get_extension()

        self._write(
            filepath=filepath,
            data=data,
        )

    @abc.abstractmethod
    def _write(self, filepath: Path, data: dict[str, np.ndarray]) -> None:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_extension() -> str:
        """
        Return the extension of the file.

        :return: Extension of the file.
        """
        pass


class NPZDataIO(DataIOBase):
    def load(self, filepath: Path) -> dict[str, np.ndarray]:
        data = np.load(
            file=filepath,
            allow_pickle=True,
        )

        return data


    def _write(self, filepath: Path, data: dict[str, np.ndarray]) -> None:
        np.savez(filepath, **data)

    @staticmethod
    def get_extension() -> str:
        return ".npz"
