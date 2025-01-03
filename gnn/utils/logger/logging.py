import logging
import logging.config
from pathlib import Path

from utils import read_json


def setup_logging(save_dir: str, log_config: str = "logger/logger_config.json",
                  default_level: logging = logging.INFO) -> None:
    """Setup logging configuration. """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)

        # Modify logging paths based on running config.
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        logging.config.dictConfig(config)
    else:
        logging.warning("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
