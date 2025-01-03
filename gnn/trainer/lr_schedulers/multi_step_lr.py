
from bisect import bisect_right
from typing import List

from gnn.trainer.lr_schedulers.base_lr import BaseLearningRate


class MultiStepLearningRate(BaseLearningRate):
    def __init__(self, lr: float = 0.001, gamma: float = 0.1, milestones: List[int] = []):
        """Multi-step learning rate scheduler.

        Args:
            optimizer: Optimizer instance.
            lr: Initial learning rate.
            gamma: Gamma value.
            milestones: Milestone to update learning rate.
        """
        super(MultiStepLearningRate, self).__init__()
        self.lr = lr
        self.initial_lr = lr
        self.gamma = gamma
        self.milestones = milestones

    def _step_lr(self, epoch: int, step: int = None) -> float:
        self.lr = self.initial_lr * self.gamma ** bisect_right(self.milestones, epoch)
        return self.lr
