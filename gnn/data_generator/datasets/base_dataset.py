from typing import Any, Dict

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """An abstract base dataset class for creating dataset instances. """

    @classmethod
    def _from_config(cls, config: Dict[str, Any], **kwargs: Dict[str, Any]) -> "BaseDataset":
        return cls(config, **kwargs)
