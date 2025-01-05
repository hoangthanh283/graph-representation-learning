from typing import Any, Dict

import munch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """An abstract base dataset class for creating dataset instances. """

    @classmethod
    def _from_config(cls, config: munch.munchify, **kwargs: Dict[str, Any]) -> "BaseDataset":
        return cls(config, **kwargs)
