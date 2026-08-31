"""
Logging setup and utility functions.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Optional
import yaml


def setup_logging(
    config_path: str | Path = "config/logging.yaml",
    log_dir: str | Path = "data",
) -> None:
    """Configures application logging from a YAML configuration file."""
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(config_path)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Returns a logger instance scoped under 'alpacha'."""
    if name:
        return logging.getLogger(f"alpacha.{name}")
    return logging.getLogger("alpacha")
