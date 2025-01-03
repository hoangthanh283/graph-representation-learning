import importlib
import logging
from datetime import datetime
from typing import Any, Dict

from tensorboardX import SummaryWriter


class TensorboardWriter:
    def __init__(self, log_dir: str, logger: logging.Logger, is_enabled: bool):
        """ To write out events and summaries to the event file.

        Args:
            log_dir: The directory of log.
            logger: Logger instance.
            is_enabled: Whether to enable logging or not.
        """
        self.writer: SummaryWriter = None
        self.selected_module: str = ""
        if is_enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False

                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on "
                          "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch "
                          "to version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the "
                          "'config.json' file."
                logger.warning(message)

        self.step = 0
        self.mode = ""
        self.tb_writer_ftns = {
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding"}
        self.tag_mode_exceptions = {"add_histogram", "add_embedding"}
        self.timer = datetime.now()

    def set_step(self, step: int, mode: str = "train") -> None:
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name: str) -> str:
        """If visualization is configured to use, return add_data() methods of tensorboard with
        additional information (step, tag) added. Otherwise, return a blank function handle that does nothing.
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag: str, data: str, *args: Dict[str, Any], **kwargs: Dict[str, Any]):
                if add_data is not None:

                    # Add mode(train/valid) tag.
                    if name not in self.tag_mode_exceptions:
                        tag = "{}/{}".format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # Default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(
                    self.selected_module, name))
            return attr
