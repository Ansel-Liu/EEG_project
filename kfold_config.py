from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch


@dataclass
class KFoldConfig:
    data_dir: Path
    output_dir: Path

    # Subject-level GroupKFold cross-validation
    kfold_multiply: bool = False
    n_splits: int = 5
    kfold_typ: str = "window" # "seizure"  # "patient" "window"
    val_subject_fraction: float = 0.20
    random_seed: int = 42

    # Training hyperparameters
    model_type: str = "cnn_lstm"  # "input_fusion","cnn_lstm","feature_fusion"
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    num_workers: int = 0
    weight_decay: float = 1e-4

    # Early stopping
    patience: int = 7

    # Model
    eeg_channels: int = 21
    signal_length: int = 128
    channel_embedding_dim: int = 64
    fusion_hidden_dim: int = 128
    out_features: int = 1
    dropout: float = 0.2

    # Imbalance handling
    use_weighted_sampler: bool = True
    use_pos_weight_loss: bool = False

    # Runtime
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for k, v in payload.items():
            if isinstance(v, Path):
                payload[k] = str(v)
            elif isinstance(v, torch.device):
                payload[k] = v.type

        return payload

    def set_kfold_multiply(self):
        self.kfold_multiply = self.kfold_typ == "window"