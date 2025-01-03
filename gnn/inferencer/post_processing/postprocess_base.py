from typing import Any, Dict


class PostProcessBase:
    """An abstract base post-process class for creating post-processing instances. """

    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "PostProcessBase":
        return cls(**config)

    def __call__(self):
        raise NotImplementedError
