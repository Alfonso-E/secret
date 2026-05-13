"""Central logging configuration.

Imports just `log` and uses `log.info(...)`, `log.warning(...)`, etc.
Output goes to BOTH the console and a rotating file at logs/bot.log.

Cloud deployments tail the log file (or `journalctl` for systemd).
Local runs see the same output on stdout as before.
"""

from __future__ import annotations

import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "bot.log"
HEARTBEAT_FILE = LOG_DIR / "heartbeat"

_FORMAT = "%(asctime)s.%(msecs)03dZ  %(levelname)-7s  %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"

# Force UTC timestamps regardless of host TZ — every cloud server should agree.
logging.Formatter.converter = time.gmtime


def _build() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(_FORMAT, _DATEFMT)

    logger = logging.getLogger("bot")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:        # idempotent across re-imports
        return logger

    # Console — always on, picks up the level from the LOG_LEVEL env var if set.
    console = logging.StreamHandler()
    console.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File — 5 x 2 MB rotating files. Keeps ~10 MB of recent history.
    fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = _build()


def heartbeat(extra: str = "") -> None:
    """Write a small file each cycle so a watchdog can detect a stalled bot.

    Cron / systemd / Docker healthchecks compare HEARTBEAT_FILE's mtime
    against now — if it hasn't been updated in ~2 hours, something is wrong.
    """
    HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_FILE.write_text(
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n{extra}\n",
        encoding="utf-8",
    )
