import abc
from pathlib import Path
import pandas as pd

class MetaDataIO(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def load(self, filepath: Path) -> pd.DataFrame:
        pass


class ParquetDataIO(MetaDataIO):
    def load(self, filepath: Path) -> pd.DataFrame:
        """
        Read a parquet file.

        :param filepath: Path to the file.
        :return: Pandas DataFrame.
        """
        df = pd.read_parquet(path=filepath)

        return df
