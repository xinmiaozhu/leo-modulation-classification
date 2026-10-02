"""Logging helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_LOGGERS: dict[str, logging.Logger] = {}


def _make_console_handler(level: int) -> logging.Handler:
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            level=level,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    except Exception:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.setLevel(level)

    return handler


def setup_logger(
    name: str = "leo_drc_dualnet",
    log_file: str | Path | None = None,
    level: int | str = logging.INFO,
    reset: bool = False,
) -> logging.Logger:
    """Create or configure a logger."""

    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if reset:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    if logger.handlers:
        return logger

    logger.addHandler(_make_console_handler(level))

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


def get_logger(name: str = "leo_drc_dualnet") -> logging.Logger:
    """Get an existing logger or create a default console logger."""

    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name=name)
    _LOGGERS[name] = logger
    return logger
