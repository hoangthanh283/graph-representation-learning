
from gnn.trainer.lr_schedulers.base_lr import BaseLearningRate


class WarmupLearningRate(BaseLearningRate):
    def __init__(self, lr: float = 0.001, warmup_lr: float = 1e-5, steps: int = 4000):
        """Piece-wise constant learning rate scheduler.

        Args:
            optimizer: Optimizer instance.
            lr: Initial learning rate.
            warmp_lr: Step tp reset learing rate.
            step: Number of steps.
        """
        super(WarmupLearningRate, self).__init__()
        self.lr = lr
        self.initial_lr = lr
        self.steps = steps
        self.warmup_learning_rate = warmup_lr

    def _step_lr(self, epoch: int, step: int = None) -> float:
        if epoch == 0 and step < self.steps:
            self.lr = self.warmup_learning_rate
        else:
            self.lr = self.initial_lr

        return self.lr
