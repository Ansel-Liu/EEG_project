from torch.utils.data import DataLoader

from data_loader.data_set import NpzDataset


def make_loader(ds: NpzDataset, batch_size: int, num_workers: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )