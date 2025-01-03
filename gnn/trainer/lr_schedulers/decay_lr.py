import numpy as np

from gnn.trainer.lr_schedulers.base_lr import BaseLearningRate


class DecayLearningRate(BaseLearningRate):
    def __init__(self, lr: float = 0.002, factor: float = 0.9, num_epochs: int = 100):
        """Decaying learning rate scheduler.

        Args:
            optimizer (cls): optimizer instance
            lr: Initial learning rate.
            factor: LR factor.
            num_epochs: Number of epochs.
        """
        super(DecayLearningRate, self).__init__()
        self.lr = lr
        self.initial_lr = lr
        self.factor = factor
        self.epochs = num_epochs

    def _step_lr(self, epoch: int, step: int = None) -> float:
        rate = np.power(1.0 - epoch / float(self.epochs + 1), self.factor)
        self.lr = self.initial_lr * rate
        return self.lr
