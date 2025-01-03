from typing import Any, Dict


class BaseAugmentor:
    """An abstract base augmentor class for creating augmentor instances. """

    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "BaseAugmentor":
        return cls(**config)

    def __call__(self):
        raise NotImplementedError
