from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from gnn.trainer.losses.base_loss import BaseLoss


class FocalLoss(BaseLoss):
    """ Multi-class Focal loss implementation.

    Args:
        gamma: Gamma value.
        weight: Weight values.
    """

    def __init__(self, gamma: float = 2.0, weight: List[float] = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Dict[str, Any]) -> torch.Tensor:
        """Getting loss value.

        Args:
            pred: Model prediction (No softmax)
            target: Ground truth label

        Returns:
            Loss value.
        """
        pred = pred.transpose(1, 2)
        logpt = F.log_softmax(pred, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss
