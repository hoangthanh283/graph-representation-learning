import munch


class BaseCollate:
    """An abstract base collate class for creating collate instances. """

    @classmethod
    def _from_config(cls, config: munch.munchify) -> "BaseCollate":
        return cls(**config)

    def __call__(self):
        raise NotImplementedError
