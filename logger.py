from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


# ANSI colors
RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
WHITE   = "\033[97m"
DIM     = "\033[2m"


class _ColourFormatter(logging.Formatter):
    """Console formatter that colourises log records by level."""

    LEVEL_COLOUR={
        logging.DEBUG: DIM +"[DEBUG]"+RESET,
        logging.INFO : GREEN + "[INFO]" + RESET,
        logging.WARNING : YELLOW + "[WARNING]"+RESET,
        logging.ERROR : RED + '[ERROR]'+RESET,
        logging.CRITICAL:RED+BOLD+'[CRITICAL]'+RESET,
    }

    def format(self,record:logging.LogRecord)-> str: 
        level_tag=self.LEVEL_COLOUR.get(record.levelno,"[?]")
        #timestamp
        ts=self.formatTime(record,"%H:%M:%S")
        ts_str=DIM+ts+RESET
        name_str=CYAN +record.name + RESET
        msg=record.getMessage()
        return f"{ts_str} {level_tag} {name_str} {msg}"




class _PlainFormatter(logging.Formatter):
    """ Plain formatter for the log file (no ANSI codes)"""
    def format(self,record:logging.LogRecord)-> str:
        ts = self.formatTime(record,"%Y-%m-%d %H:%M:%S")
        return f"{ts} [{record.levelname:<8}] {record.name}{record.getMessage()}"

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
        log_path/'research_agent.log',
        maxBytes=5 *1024*1024,
        backupCount=3,
        encoding='utf-8'
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_PlainFormatter())
    logger.addHandler(fh)

    _INITIALISED.add(name)
    return logger



def phase_banner(logger:logging.Logger,phase_num:int,title:str)->None:
    """Print a bold phase header to the console"""
    bar = "="*60
    logger.info(f"\n{BOLD}{CYAN}{bar}{RESET}")
    logger.info(f'{BOLD}{CYAN} PHASE {phase_num} -- {title.upper()}{RESET}')
    logger.info(f'{BOLD}{CYAN}{bar}{RESET}')

def step(logger : logging.Logger, msg:str)-> None:
    """Print a step line"""
    logger.info(f"{GREEN} {RESET} {msg}")

def substep(logger:logging.Logger,msg:str)-> None:
    """Print an indented sub_step line."""
    logger.info(f'{DIM} ->{RESET}{msg}')

def warn(logger:logging.Logger,msg:str)-> None:
    """Print an yellow warning."""
    logger.warning(f"{YELLOW} {RESET}{msg}")

def success(logger:logging.Logger,msg:str)-> None:
    """Print a bold green success line."""
    logger.info(f'{BOLD}{GREEN} {msg}{RESET}')

def section_title(logger:logging.Logger,title:str)-> None:
    """Print a dimmed section divider."""
    logger.info(f'{DIM}{"-"*50}{RESET}')
    logger.info(f"{BOLD}{WHITE} {title}{RESET}")

