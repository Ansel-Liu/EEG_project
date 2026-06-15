from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader.data_set import NpzDataset
from kfold_config import KFoldConfig
from data_utils import ensure_dir, set_seed, get_kfold_split, describe_split, make_loader, calculate_pos_weight, \
    save_json
from models import get_criterion_optimizer
from provider import get_dataset_split, get_model
from training import fit, evaluate


def kfold_training(
        cfg: KFoldConfig,
):
    ensure_dir(cfg.output_dir)
    set_seed(cfg.random_seed)

    print("Using device:", cfg.device)
    print(
        f"Training settings -> n_splits={cfg.n_splits}, batch_size={cfg.batch_size}, "
        f"epochs={cfg.epochs}, lr={cfg.lr}, early_stopping_patience={cfg.patience}"
    )

    all_fold_metrics = []
    fold_summaries = []

    fold_defs = get_kfold_split(
        data_directory=cfg.data_dir,
        n_splits=cfg.n_splits,
        random_seed=cfg.random_seed,
        multiply=cfg.kfold_multiply,
    )

    for fold_idx, fold_def in enumerate(fold_defs):
        fold_name = f"fold_{fold_idx}"
        print("\n" + "=" * 70)
        print(f"Running {fold_name}")
        print("=" * 70)

        train_split, val_split, test_split = get_dataset_split(
            typ=cfg.kfold_typ,
            data_directory=cfg.data_dir,
            pat_id_test=fold_def,
            val_split=cfg.val_subject_fraction,
        )

        print("===== SPLIT SUMMARY =====")
        train_summary = describe_split(split=train_split, name=f"{fold_name.upper()} TRAIN")
        val_summary = describe_split(split=val_split, name=f"{fold_name.upper()} VAL")
        test_summary = describe_split(split=test_split, name=f"{fold_name.upper()} TEST")

        train_ds = NpzDataset(**train_split)
        val_ds = NpzDataset(**val_split)
        test_ds = NpzDataset(**test_split)


        train_loader = make_loader(ds=train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
        val_loader = make_loader(ds=val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        test_loader = make_loader(ds=test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

        model = get_model(typ=cfg.model_type, eeg_channels=cfg.eeg_channels, dropout=cfg.dropout).to(cfg.device)
        pos_weight = calculate_pos_weight(train_loader=train_loader)
        #0409 by wu
        if isinstance(pos_weight, torch.Tensor):
            pos_weight = pos_weight.to(cfg.device)
        #0409 by wu
        criterion, optimizer = get_criterion_optimizer(pos_weight=pos_weight, model=model, learning_rate=cfg.lr)

        model, history_df = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=cfg.device,
            epochs=cfg.epochs,
            patience=cfg.patience,
            fold_name=fold_name,
        )

        test_loss, test_metrics, y_true_test, y_prob_test = evaluate(model, test_loader, criterion, cfg.device)
        print(f"[{fold_name}] test_loss={test_loss:.4f}")
        for k, v in test_metrics.items():
            print(f"[{fold_name}] {k}: {v:.4f}")

        fold_dir = cfg.output_dir / fold_name
        ensure_dir(fold_dir)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": cfg.dict(),
                "fold": fold_idx,
                "train_patients": train_summary["patient_ids"],
                "val_patients": val_summary["patient_ids"],
                "test_patients": test_summary["patient_ids"],
            },
            fold_dir / "model.pt"
        )

        history_df.to_csv(fold_dir / "history.csv", index=False)

        np.savez(
            file=fold_dir / "test_predictions.npz",
            y_true=y_true_test,
            y_pred=y_prob_test,
            **{k: v for k, v in test_split.items() if k not in ["X", "y"]},
        )

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
    metrics_df.to_csv(cfg.output_dir / "crossval_metrics.csv", index=False)

    summary_row = {"fold": "mean"}
    for col in [c for c in metrics_df.columns if c != "fold"]:
        summary_row[col] = metrics_df[col].mean()
    std_row = {"fold": "std"}
    for col in [c for c in metrics_df.columns if c != "fold"]:
        std_row[col] = metrics_df[col].std(ddof=1)
    summary_df = pd.DataFrame([summary_row, std_row])
    summary_df.to_csv(cfg.output_dir / "crossval_summary.csv", index=False)

    save_json(cfg.output_dir / "config.json", cfg.dict())
    save_json(cfg.output_dir / "fold_summaries.json", fold_summaries)

    print("\n===== CROSS-VALIDATION RESULTS =====")
    print(metrics_df)
    print("\nMean metrics:")
    print(summary_df.iloc[[0]])
    print(f"\nSaved outputs to: {cfg.output_dir}")


if __name__ == "__main__":
    for preprocessing in ["Preprocessed_median", "Preprocessed"]:
        preprocess_path = Path("dataset", preprocessing)
    for kfold_type in ["seizure", "patient", "window"]:
        for model_type in ["feature_fusion", "input_fusion", "cnn_lstm"]:
            # ToDo delete
            if model_type == "cnn_lstm" and kfold_type != "window":
                print(f"Skipping {kfold_type}, {model_type}")
                continue
            if model_type in ["feature_fusion", "input_fusion"] and kfold_type != "window" and preprocessing == "Preprocessed":
                print(f"Skipping {kfold_type}, {model_type} for {preprocessing}")
                continue

            if preprocessing == "Preprocessed_median":
                out_path = Path("kfold_output", f"{kfold_type}_{model_type}_mediansmoothing")
            else:
                out_path = Path("kfold_output", f"{kfold_type}_{model_type}")

            print("="*40, f"KFold with type {kfold_type} and model {model_type} start for {preprocessing}", "="*40)

            config = KFoldConfig(
                data_dir=preprocess_path,
                output_dir=out_path,
                kfold_typ=kfold_type,
                model_type=model_type,
            )
            config.set_kfold_multiply()

            kfold_training(cfg=config)

            print("="*40, f"KFold with type {kfold_type} and model {model_type} end for {preprocessing}", "="*40)
