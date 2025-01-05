import munch


class BaseAugmentor:
    """An abstract base augmentor class for creating augmentor instances. """

    @classmethod
    def _from_config(cls, config: munch.munchify) -> "BaseAugmentor":
        return cls(**config)

    def __call__(self):
        raise NotImplementedError
