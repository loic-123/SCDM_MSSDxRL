"""Structured logging setup for MSSD."""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_dir: str = "artifacts/logs", level: str = "INFO"
) -> logging.Logger:
    """Set up structured logging to both file and console."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"mssd_{timestamp}.log"

    logger = logging.getLogger("mssd")
    logger.setLevel(getattr(logging, level))

    # Avoid adding duplicate handlers
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(ch)

        fh = logging.FileHandler(log_file)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
            )
        )
        logger.addHandler(fh)

    return logger
