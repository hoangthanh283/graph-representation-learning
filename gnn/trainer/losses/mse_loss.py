from typing import Any, Dict

import torch

from gnn.trainer.losses.base_loss import BaseLoss


class MSELoss(BaseLoss):
    def __init__(self):
        """Constructs the cross_entropy loss, weighted cross_entropy or dice_coefficient.

        Args:
            weight: Weights for the different classes in case of multi-class imbalance.
        """
        super(MSELoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Dict[str, Any]) -> torch.Tensor:
        """Getting loss value.

        Args:
            pred: Model prediction (No softmax)
            target: Ground truth label

        Returns:
            Loss value.
        """
        mask = target != -100
        loss = torch.sum(
            ((pred.reshape(target.shape) - target) * mask) ** 2
        ) / torch.sum(mask)
        return loss
