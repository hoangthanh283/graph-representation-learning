from typing import Any, Dict

import torch

from gnn.trainer.losses.base_loss import BaseLoss


class BinaryCrossEntropyLoss(BaseLoss):
    def __init__(self, pos_weight: torch.Tensor = None):
        """Constructs the cross_entropy loss, weighted cross_entropy or dice_coefficient.

        Args:
            weight: Weights for the different classes in case of multi-class imbalance.
        """
        super(BinaryCrossEntropyLoss, self).__init__()
        if pos_weight is not None:
            self.criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=pos_weight, reduction="none"
            )
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Dict[str, Any]) -> torch.Tensor:
        """Getting loss value.

        Args:
            pred: Model prediction (No softmax)
            target: Ground truth label

        Returns:
            Loss value.
        """
        mask = target != -100
        loss = self.criterion(pred.reshape(target.shape), target)
        loss = loss * mask
        loss = torch.sum(loss) / torch.sum(mask)

        return loss
