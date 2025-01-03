from typing import Any, Dict


class BaseDataProcess:
    """AN abstract base processing class for creating processing instances. """

    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "BaseDataProcess":
        return cls(**config)

    def __call__(self):
        raise NotImplementedError
