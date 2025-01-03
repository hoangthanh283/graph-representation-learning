from typing import Any, Dict

import torch.nn as nn

from gnn import models
from gnn.utils.logger.color_logger import color_logger


class BaseNetwork(nn.Module):
    def __init__(self):
        """Base network class which all network inherits. """
        super(BaseNetwork, self).__init__()
        self.logger = color_logger(__name__, testing_mode=False)

    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "BaseNetwork":
        """Creates specified Model object from a given config dict.

        Args:
            config: Dictionary that contains model config.
                Expected fields:
                    model: Model class name.
                    args: Arguments for model constructor.
            # TODO: Remove region_presidio dependency from TemplateEngine.

        Returns:
            BaseModel instance.

        Raises:
            KeyError: In case of missing config fields.
            TypeError: In case of missing or wrong keyword arguments defined in `args`.
        """
        model_class = getattr(models, config["type"], None)
        if model_class is None:
            raise KeyError(f"Cannot find {config['type']} class. "
                           f"Make sure to import this class in {models.__name__}.__init__.py.")
        if cls is BaseNetwork:
            return model_class._from_config(config)

        args = config.get("args", {})
        try:
            model = model_class(**args)
            model.logger.info(f"Num parameters of {model.__class__.__name__}: {model._count_parameters()}")
            return model
        except TypeError as er:
            raise TypeError(f"{er}. Check `args` fields defined in config with the actual keyword args "
                            f"required in {model_class.__name__} `__init__` method.")

    def _count_parameters(self) -> str:
        """Counting number of model parameters. """
        num_params = sum(pp.numel() for pp in self.parameters() if pp.requires_grad)
        return f"{num_params:,}"

    def forward(self):
        """Forwarding process. """
        raise NotImplementedError
