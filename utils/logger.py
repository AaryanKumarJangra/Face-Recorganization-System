"""
logger.py
=========

Centralized logger factory. Every module gets its logger by calling
`get_logger(__name__)` — this guarantees consistent formatting and ensures
logs are written to both console and per-run log files under logs/,
as required by the project spec (separate logs for training, recognition,
errors).

Usage
-----
    from utils.logger import get_logger

    logger = get_logger(__name__, log_filename="training.log")
    logger.info("Starting training...")
    logger.error("Something went wrong", exc_info=True)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Tracks which loggers have already been configured so repeated calls to
# get_logger() for the same name don't attach duplicate handlers.
_CONFIGURED_LOGGERS: set[str] = set()


def get_logger(
    name: str,
    log_dir: str = "logs",
    log_filename: Optional[str] = None,
    level: str = "INFO",
    log_to_console: bool = True,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Return a configured logger. Safe to call multiple times with the same
    `name` — handlers are only attached once per logger name.

    Parameters
    ----------
    name : str
        Usually `__name__` of the calling module.
    log_dir : str
        Directory to write log files into (created if missing).
    log_filename : Optional[str]
        File to write this logger's records to. If None, defaults to
        "<name>.log". Use a shared name (e.g. "training.log") across
        modules in the same phase to consolidate logs.
    level : str
        One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    log_to_console : bool
        Whether to also stream logs to stdout.
    log_to_file : bool
        Whether to persist logs to a file under log_dir.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # avoid duplicate logs via the root logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        filename = log_filename or f"{name.split('.')[-1]}.log"
        file_handler = logging.FileHandler(log_path / filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger
