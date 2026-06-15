import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam


def get_criterion_optimizer(pos_weight: torch.Tensor, model: nn.Module, learning_rate: float) -> tuple[
    BCEWithLogitsLoss, Adam]:
    criterion = BCEWithLogitsLoss(weight=pos_weight)
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
    )

    return criterion, optimizer