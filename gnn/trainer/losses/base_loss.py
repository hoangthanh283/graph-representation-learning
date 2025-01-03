from typing import Any, Dict

import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self):
        """Base loss class. Specifies the interface used by different loss types. """
        super(BaseLoss, self).__init__()

    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "BaseLoss":
        return cls(**config)

    def forward(self):
        raise NotImplementedError
