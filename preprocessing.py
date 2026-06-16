from pathlib import Path

from tqdm import tqdm

from data_utils import (
    get_scaler,
    label_data,
)
from io_data import DataLoader, NPZDataIO, ParquetDataIO, DataIOBase
from scipy.ndimage import gaussian_filter1d, median_filter
import numpy as np

def preprocessing(
        raw_directory: Path,
        data_loader: DataLoader,
        out_directory: Path,
        data_io: DataIOBase,
        config: dict,
):
    raw_paths = list(raw_directory.glob(f"*{data_loader.get_extension()}"))

    print("="*30, "Start Preprocessing", "="*30)
    for raw_path in tqdm(raw_paths):
        data, meta = data_loader.load_eeg_window_data_and_meta(
            eeg_window_data_filepath=raw_path,
        )
        #0409
        data = np.asarray(data).astype(np.float32)
        if "smoothing" in config and config["smoothing"].get("use", False):
            smooth_type = config["smoothing"].get("type", "median")
            if smooth_type == "gaussian":
                sigma = float(config["smoothing"].get("sigma", 1.0))
                data = gaussian_filter1d(data, sigma=sigma, axis=-1)
                
            elif smooth_type == "median":
                kernel_size = int(config["smoothing"].get("kernel_size", 5))
                
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                kernel_sizes = [1] * data.ndim
                kernel_sizes[-1] = kernel_size
                
                data = median_filter(data, size=tuple(kernel_sizes))
        #0409
        if "scaling" in config:
            if config["scaling"].get("use", False):
                scaler = get_scaler(
                    scaler_name=config["scaling"].get("scaler_name", "undefined"),
                )
                data = scaler.scale_X(X=data)


        labeled_data = label_data(
            data=data,
            metadata=meta,
        )

        data_io.write(
            filepath=out_directory / f"{raw_path.name}",
            data=labeled_data,
        )

    print("="*30, "Finished Preprocessing", "="*30)

if __name__ == "__main__":
    _config = {
        "smoothing": {
            "use": True,
            "type": "median",
            "kernel_size": 3,
            "sigma": 1.0
        },

        "scaling": {
            "use": True,
            "scaler_name": "standardTransposed",
        }
    }
    _raw_directory = Path("dataset", "Epilepsy")
    npz_loader = NPZDataIO()
    parquet_loader = ParquetDataIO()
    _data_loader = DataLoader(
        raw_data_loader=npz_loader,
        meta_data_loader=parquet_loader,
    )
    _out_directory = Path("dataset", "Preprocessed_median")#Preprocessed_median,Preprocessed
    preprocessing(
        raw_directory=_raw_directory,
        data_loader=_data_loader,
        out_directory=_out_directory,
        data_io=npz_loader,
        config=_config,
    )