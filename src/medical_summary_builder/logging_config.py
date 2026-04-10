"""Logging setup for Medical Summary Builder.

Call ``setup_logging(log_path)`` once at the start of a run.  It wires up:
- A rotating file handler that writes DEBUG-and-above to ``logs/<stem>.log``.
- A stream handler that only emits WARNING-and-above, so the rich console
  output stays clean while the file captures every detail.
"""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_path: Path) -> Path:
    """Configure root logger to write to *log_path* (created if needed).

    Returns the resolved log path so callers can display it.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any handlers already attached (e.g. from previous test runs)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — captures DEBUG and above
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler — only WARNING and above (rich handles normal output)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    logging.getLogger(__name__).info("Logging initialised → %s", log_path)
    return log_path
