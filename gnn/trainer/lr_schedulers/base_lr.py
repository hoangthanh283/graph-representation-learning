import munch


class BaseLearningRate:
    """Base learning rate scheduler. """
    @classmethod
    def _from_config(cls, config: munch.munchify) -> "BaseLearningRate":
        return cls(**config)

    def _step_lr(self):
        raise NotImplementedError
