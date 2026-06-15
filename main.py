from pathlib import Path

from io_data import (
    DataLoader,
    NPZDataIO,
    ParquetDataIO,
)


if __name__ == '__main__':
    data_loader = DataLoader(
        raw_data_loader=NPZDataIO(),
        meta_data_loader=ParquetDataIO(),
    )

    #filepath = Path("dataset", "Epilepsy", "chb01_seizure_EEGwindow_1.npz")
    #data, meta = data_loader.load_eeg_window_data_and_meta(eeg_window_data_filepath=filepath)
    filepath = Path("dataset", "Epilepsy", "chb01_seizure_metadata_1.parquet")
    meta = ParquetDataIO().load(filepath=filepath)

"""if __name__ == "__main__":
    import numpy as np

    L = np.array([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1]]).astype(int)
    G = np.array([0,0,1,1,1,0,0,0,0,0,2,2,2,2,2,1,1,1,2,2,2]).astype(int)

    test_mask = np.zeros(L.shape[0], dtype=bool)
    train_mask = np.ones(L.shape[0], dtype=bool)

    for l in [0, 1]:
        # alle Positionen mit Label 1
        ones_idx = np.where(L == l)[0]

        # alle Positionen, wo L==1 (für spätere Zuordnung)
        g_at_ones = G[ones_idx]

        # einzigartige globale Indizes, die überhaupt Label=1 haben
        unique_g = np.unique(g_at_ones)

        # --- letzter Index pro global index ---
        # wir nehmen für jedes g das letzte Auftreten in ones_idx
        mask = g_at_ones[:, None] == unique_g[None, :]
        last_pos_in_ones = mask.cumsum(axis=0).argmax(axis=0)

        last_indices = ones_idx[last_pos_in_ones]
        test_mask[last_indices] = True
        train_mask[last_indices] = False

        if l == 1:
            # --- die 4 vorherigen ---
            # Positionen in ones_idx nehmen und 4 zurückgehen
            offsets = np.arange(1, 5)

            prev_indices = last_pos_in_ones[:, None] - offsets[None, :]
            prev_indices = np.clip(prev_indices, 0, len(ones_idx) - 1)

            prev_four = ones_idx[prev_indices]
            prev_four_unique = np.unique(prev_four)
            train_mask[prev_four_unique] = False

    print(test_mask)
    print(train_mask)"""