from typing import Any, Dict


class BaseOptimizer:
    """Base optimizer. """
    @classmethod
    def _from_config(cls, opt: Dict[str, Any]) -> "BaseOptimizer":
        return cls(**opt)

    def get_optimizer(self):
        raise NotImplementedError
