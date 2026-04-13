from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path



# Module-level singleton registry 
_INITIALISED : set[str]=set()

def get_logger(name:str='research_agent',log_dir:str| Path='logs')->logging.Logger:
    """
    Return (and lazily initialise) a named logger.

    Parameters
    ----------
    name    : logger name, usually the module __name__
    log_dir : directory for the rotating log file (created if absent)

    Usage
    -----
    >>> from phd_research_agent.logger import get_logger
    >>> log = get_logger(__name__)
    >>> log.info("Hello from my module")
    """
    logger=logging.getLogger(name)

    # only configure once 
    if name in _INITIALISED:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate=False

    #console handler
    ch=logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ColourFormatter())
    logger.addHandler(ch)

    #file handler
    log_path=Path(log_dir)
    log_path.mkdir(parents=True,exist_ok=True)
    fh= RotatingFileHandler(
        log_path/'research_agent.log'
        maxBytes=5 *1024*1024
        backupCount=3,
        encoding='utf-8'
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_PlainFormatter())
    logger.addHandler(fh)

    _INITIALISED.add(name)
    return logger