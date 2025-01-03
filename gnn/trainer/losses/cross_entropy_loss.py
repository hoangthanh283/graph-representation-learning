from typing import Any, Dict, List

import numpy as np
import torch

from gnn.trainer.losses.base_loss import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __init__(self, weight: List[float] = None):
        """ Constructs the cross_entropy loss, weighted cross_entropy or dice_coefficient.

        Args:
            weight: Weights for the different classes in case of multi-class imbalance.
        """
        super(CrossEntropyLoss, self).__init__()
        if weight is not None:
            weight = torch.from_numpy(np.array(weight, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Dict[str, Any]) -> torch.Tensor:
        """Getting loss value.

        Args:
            pred: Model prediction (No softmax)
            target: Ground truth label

        Returns:
            Loss value.
        """
        pred = pred.transpose(1, 2)
        loss = self.criterion(pred, target)
        return loss
