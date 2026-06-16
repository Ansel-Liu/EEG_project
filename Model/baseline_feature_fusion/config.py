from dataclasses import dataclass


@dataclass
class Config:
    data_dir: str = "/hhome/ricse06/deppLearning/UAB-Deep-Learning-Epilepsy/dataset/Epilepsy"

    # Subject-level GroupKFold cross-validation
    n_splits: int = 5
    val_subject_fraction: float =    0.20
    random_seed: int = 42

    # Training hyperparameters
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
    dropout: float = 0.3

    # Imbalance handling
    use_weighted_sampler: bool = True
    use_pos_weight_loss: bool = False

    # Runtime
    device: str = "cuda"
    output_dir: str = "artifacts_groupkfold"
