from typing import Any, Dict


class BaseLearningRate:
    """Base learning rate scheduler. """
    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "BaseLearningRate":
        return cls(**config)

    def _step_lr(self):
        raise NotImplementedError
