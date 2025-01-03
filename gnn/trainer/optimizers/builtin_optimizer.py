from typing import Any, Dict

import torch
import torch.nn as nn

from gnn.trainer.optimizers.base_optimzer import BaseOptimizer


class BuitlinOptimizer(BaseOptimizer):
    def __init__(self, type_optimizer: str, lr: float, **kwargs: Dict[str, Any]):
        """Create default optimizers that are availabel in pytorch from the config.

        Args:
            type_optimizer: TYpe of the optimizer.
            lr: Initial learning rate.
        """
        super(BuitlinOptimizer, self).__init__()
        self.learning_rate = lr
        self.type_optimizer = type_optimizer
        self.optimizer_args = kwargs

    def get_optimizer(self, parameters: nn.Parameter) -> torch.optim:
        optimizer = getattr(torch.optim, self.type_optimizer)(parameters, **self.optimizer_args)
        if hasattr(self.learning_rate, "prepare"):
            self.learning_rate.prepare(optimizer)
        return optimizer
