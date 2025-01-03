import os
from typing import Any, Dict

import torch

from gnn.utils.logger.color_logger import color_logger


class CheckpointHandler:
    def __init__(self):
        """Saving & loading checkpoints. """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.logger = color_logger(__name__, testing_mode=False)
        self.logger.info("Initializing checkpoint handler ...")

    def save_checkpoint(self, checkpoint: Dict[str, Any], output_path: str,
                        epoch: int = None, step: int = None) -> None:
        """Saving checkpoint with given outoutput path, epoch, and step.

        Args:
            Checkpoint: Checkpoint dictionary with model states & other meta data.
            output_path: Path to save checkpoint.
            epoch: Ith epoch.
            step: Ith step.
        """
        checkpoint_name = self.make_checkpoint_name("model", epoch, step)
        os.makedirs(output_path, exist_ok=True)
        torch.save(checkpoint, os.path.join(output_path, checkpoint_name))
        self.logger.info("Saving checkpoint success!")

    def make_checkpoint_name(self, name: str, epoch: int = None, step: int = None) -> str:
        """Generate docstring name for checkpoint.

        Args:
            name: Name of checkpoint.
            epoch: Ith epoch.
            step: Ith step.

        Returns:
            Docstring of checkpoint name.
        """
        if epoch is None or step is None:
            c_name = name + "_latest.pt"
        else:
            c_name = f"{name}_epoch_{epoch}_minibatch_{step}.pt"
        return c_name

    def restore_checkpoint(self, checkpoint_path: str = None) -> Dict[str, Any]:
        """Load model weight & previous configurations.

        Args:
            model_path: Path to model weight.

        Return:
            model: Model instance.
        """
        checkpoint_dict = torch.load(checkpoint_path, map_location=self.device)
        self.logger.info("Loading checkpoint success!")
        return checkpoint_dict
