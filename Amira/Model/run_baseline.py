import argparse
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baseline_feature_fusion.config import Config
from baseline_feature_fusion.data import inspect_data, load_all_patients
from baseline_feature_fusion.dataset import (
    EEGWindowDataset,
    compute_channel_stats,
    make_weighted_sampler,
    normalize_eeg,
)
from baseline_feature_fusion.model import EEGFeatureFusionCNN
from baseline_feature_fusion.splits import (
    describe_split,
    make_subject_groupkfold_splits,
    make_subject_val_split,
)
from baseline_feature_fusion.train import evaluate, fit
from baseline_feature_fusion.utils import ensure_dir, save_json, set_seed


def build_model_and_criterion(cfg: Config, device, y_train: np.ndarray):
    model = EEGFeatureFusionCNN(
        eeg_channels=cfg.eeg_channels,
        emb_dim=cfg.channel_embedding_dim,
        fusion_hidden_dim=cfg.fusion_hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    if cfg.use_pos_weight_loss:
        pos_count = float(y_train.sum())
        neg_count = float(len(y_train) - pos_count)
        pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    return model, criterion, optimizer


def make_loader(dataset, batch_size, num_workers, sampler=None, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        num_workers=num_workers,
    )


def main():
    defaults = Config()

    parser = argparse.ArgumentParser(
        description="Train EEG seizure baseline with subject-level GroupKFold cross-validation."
    )
    parser.add_argument("--data_dir", type=str, default=defaults.data_dir)
    parser.add_argument("--n_splits", type=int, default=defaults.n_splits)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--device", type=str, default=defaults.device)
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )

    ensure_dir(cfg.output_dir)
    set_seed(cfg.random_seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(
        f"Training settings -> n_splits={cfg.n_splits}, batch_size={cfg.batch_size}, "
        f"epochs={cfg.epochs}, lr={cfg.lr}, early_stopping_patience={cfg.patience}"
    )

    X, metadata, pairs = load_all_patients(cfg.data_dir)
    inspect_data(X, metadata)

    if X.shape[1] != cfg.eeg_channels or X.shape[2] != cfg.signal_length:
        raise ValueError(
            f"Expected windows of shape [N,{cfg.eeg_channels},{cfg.signal_length}], got {X.shape}"
        )

    fold_defs = make_subject_groupkfold_splits(metadata, n_splits=cfg.n_splits)
    all_fold_metrics = []
    all_fold_predictions = []
    fold_summaries = []

    for fold_def in fold_defs:
        fold_idx = fold_def["fold"]
        fold_name = f"fold_{fold_idx}"
        print("\n" + "=" * 70)
        print(f"Running {fold_name}")
        print("=" * 70)

        train_val_idx = fold_def["train_val_idx"]
        test_idx = fold_def["test_idx"]
        train_idx, val_idx = make_subject_val_split(
            metadata_subset=metadata.iloc[train_val_idx],
            val_fraction=cfg.val_subject_fraction,
            seed=cfg.random_seed + fold_idx,
        )

        print("===== SPLIT SUMMARY =====")
        train_summary = describe_split(metadata, f"{fold_name.upper()} TRAIN", train_idx)
        val_summary = describe_split(metadata, f"{fold_name.upper()} VAL", val_idx)
        test_summary = describe_split(metadata, f"{fold_name.upper()} TEST", test_idx)

        X_train = X[train_idx]
        X_val = X[val_idx]
        X_test = X[test_idx]

        y_train = metadata.iloc[train_idx]["class"].to_numpy(dtype=np.float32)
        y_val = metadata.iloc[val_idx]["class"].to_numpy(dtype=np.float32)
        y_test = metadata.iloc[test_idx]["class"].to_numpy(dtype=np.float32)

        mean, std = compute_channel_stats(X_train)
        X_train = normalize_eeg(X_train, mean, std)
        X_val = normalize_eeg(X_val, mean, std)
        X_test = normalize_eeg(X_test, mean, std)

        train_ds = EEGWindowDataset(X_train, y_train)
        val_ds = EEGWindowDataset(X_val, y_val)
        test_ds = EEGWindowDataset(X_test, y_test)

        sampler = make_weighted_sampler(y_train) if cfg.use_weighted_sampler else None
        train_loader = make_loader(train_ds, cfg.batch_size, cfg.num_workers, sampler=sampler, shuffle=True)
        val_loader = make_loader(val_ds, cfg.batch_size, cfg.num_workers, shuffle=False)
        test_loader = make_loader(test_ds, cfg.batch_size, cfg.num_workers, shuffle=False)

        model, criterion, optimizer = build_model_and_criterion(cfg, device, y_train)

        model, history_df = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=cfg.epochs,
            patience=cfg.patience,
            fold_name=fold_name,
        )

        test_loss, test_metrics, y_true_test, y_prob_test = evaluate(model, test_loader, criterion, device)
        print(f"[{fold_name}] test_loss={test_loss:.4f}")
        for k, v in test_metrics.items():
            print(f"[{fold_name}] {k}: {v:.4f}")

        fold_dir = os.path.join(cfg.output_dir, fold_name)
        ensure_dir(fold_dir)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "mean": mean,
                "std": std,
                "config": asdict(cfg),
                "fold": fold_idx,
                "patients_loaded": [p["patient_id"] for p in pairs],
                "train_patients": train_summary["patient_ids"],
                "val_patients": val_summary["patient_ids"],
                "test_patients": test_summary["patient_ids"],
            },
            os.path.join(fold_dir, "model.pt"),
        )
        history_df.to_csv(os.path.join(fold_dir, "history.csv"), index=False)

        pred_df = metadata.iloc[test_idx].copy()
        pred_df["fold"] = fold_idx
        pred_df["y_true"] = y_true_test.astype(int)
        pred_df["y_prob"] = y_prob_test
        pred_df["y_pred"] = (pred_df["y_prob"] >= 0.5).astype(int)
        pred_df.to_csv(os.path.join(fold_dir, "test_predictions.csv"), index=False)
        all_fold_predictions.append(pred_df)

        metric_row = {"fold": fold_idx, "test_loss": test_loss, **test_metrics}
        all_fold_metrics.append(metric_row)
        fold_summaries.append(
            {
                "fold": fold_idx,
                "train": train_summary,
                "val": val_summary,
                "test": test_summary,
            }
        )

    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(os.path.join(cfg.output_dir, "crossval_metrics.csv"), index=False)

    summary_row = {"fold": "mean"}
    for col in [c for c in metrics_df.columns if c != "fold"]:
        summary_row[col] = metrics_df[col].mean()
    std_row = {"fold": "std"}
    for col in [c for c in metrics_df.columns if c != "fold"]:
        std_row[col] = metrics_df[col].std(ddof=1)
    summary_df = pd.DataFrame([summary_row, std_row])
    summary_df.to_csv(os.path.join(cfg.output_dir, "crossval_summary.csv"), index=False)

    all_predictions_df = pd.concat(all_fold_predictions, axis=0, ignore_index=True)
    all_predictions_df.to_csv(os.path.join(cfg.output_dir, "all_test_predictions.csv"), index=False)

    save_json(os.path.join(cfg.output_dir, "config.json"), asdict(cfg))
    save_json(os.path.join(cfg.output_dir, "fold_summaries.json"), fold_summaries)

    print("\n===== CROSS-VALIDATION RESULTS =====")
    print(metrics_df)
    print("\nMean metrics:")
    print(summary_df.iloc[[0]])
    print(f"\nSaved outputs to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
