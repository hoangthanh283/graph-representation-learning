import logging
import os

import colorlog
from decouple import config as env_config


def color_logger(dunder_name: str, testing_mode: bool) -> logging.Logger:
    """Define color format for logging.

    Args:
        dunder_name: Value to log.
        testing_mode: Set level of logging.

    Returns:
        A color logger instance.
    """
    log_format = (
        "%(asctime)s - "
        "%(name)s - "
        "%(funcName)s - "
        "%(levelname)s - "
        "%(message)s"
    )
    colorlog_format = (
        "%(log_color)s "
        f"{log_format}"
    )
    colorlog.basicConfig(format=colorlog_format)
    logger = logging.getLogger(dunder_name)
    if testing_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Output full log.
    log_dir = env_config("OUTPUT_DIR", "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    fh = logging.FileHandler(os.path.join(log_dir, "output.log"))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Output warning log.
    fh = logging.FileHandler(os.path.join(log_dir, "output.warning.log"))
    fh.setLevel(logging.WARNING)
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Output error log.
    fh = logging.FileHandler(os.path.join(log_dir, "output.error.log"))
    fh.setLevel(logging.ERROR)
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
