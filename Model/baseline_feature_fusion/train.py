from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from .metrics import compute_metrics


def train_one_epoch(model, loader, optimizer, criterion, device) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    y_true_all = []
    y_prob_all = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_prob_all.extend(probs.tolist())
        y_true_all.extend(y.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, compute_metrics(y_true_all, y_prob_all)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true_all = []
    y_prob_all = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_prob_all.extend(probs.tolist())
        y_true_all.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(y_true_all, y_prob_all)
    return avg_loss, metrics, np.array(y_true_all), np.array(y_prob_all)


def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs, patience, fold_name="fold"):
    best_val_f1 = -1.0
    best_state = None
    history = []
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"[{fold_name}] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_f1={train_metrics['f1']:.4f} | val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[{fold_name}] Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history)
