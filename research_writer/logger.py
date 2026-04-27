from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

RESET= "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"

GREEN = "\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
MAGENTA="\033[95m"
BLUE="\033[94m"
WHITE="\033[97m"
DIM="\033[2m"

class _ColourFormatter(logging.Formatter):
    LEVEL_COLOURS={
        logging.DEBUG: DIM + "[DEBUG]" + RESET,
        logging.INFO : GREEN + "[INFO]" + RESET,
        logging.WARNING : YELLOW +"[WARNING]" + RESET,
        logging.ERROR : RED + "[ERROR]" + RESET,
        logging.CRITICAL : RED+BOLD + "[CRITICAL]" + RESET,
    }
    def format(self,record : logging.LogRecord) -> str:
        tag = self.LEVEL_COLOURS.get(record.levelno,"[?]")
        ts = DIM + self.formatTime(record,"%H:%M:%S")+RESET
        nm = MAGENTA + record.name +RESET
        return f"{ts} {tag} {nm} {record.getMessage()}"

class _PlainFormatter(logging.Formatter):
    def format(self,record :logging.LogRecord)-> str:
        ts = self.formatTime(record , "%H:%M:%S")
        return f"{ts}  [{record.levelname:<8}]  {record.name}  {record.getMessage()}"
    
_INITIALISED : set[str]=set()

def get_logger(name :str ="research_writer",log_dir:str | Path = "logs")-> logging.Logger:
    """Return (and lazily configure) a named logger"""
    logger = logging.getLogger(name)
    if name in _INITIALISED:
        return logger
    
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ColourFormatter())
    logger.addHandler(ch)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True,exist_ok=True)
    fh=RotatingFileHandler(
        log_path / "research_writer.log",
        maxBytes=5*1024*1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_PlainFormatter())
    logger.addHandler(fh)

    _INITIALISED.add(name)
    return logger


# Banner HEelper

def phase_banner(logger : logging.Logger,num:int,title:str)-> None:
    bar = "=" * 60
    logger.info(f"\n{BOLD}{MAGENTA}{bar}{RESET}")
    logger.info(f"{BOLD}{MAGENTA} STAGE {num} -- {title.upper()}{RESET}")
    logger.info(f"{BOLD} {MAGENTA} {bar}{RESET}")

def debate_banner(logger : logging.Logger , round_num:int , cluster:str)->None:
    logger.info(f"\n{BOLD}{BLUE} DEBATE Round{round_num} | Cluster : {cluster[:50]} {RESET}")

def step(logger:logging.Logger,msg :str)-> None:
    logger.info(f"{GREEN } {RESET} {msg}")

def substep(logger : logging.Logger,msg :str)-> None:
    logger.info(f"{DIM} ->{RESET} {msg}")

def warn(logger:logging.Logger,msg :str)-> None:
    logger.warning(f"{YELLOW} {RESET} {msg}")

def success(logger :logging.Logger,msg :str)-> None:
    logger.info(f"{BOLD}{GREEN}   ✅ {msg}{RESET}")

def writer_says(logger :logging.Logger,preview:str)->None:
    logger.info(f"{CYAN}  WRITER -> {preview[:80]}...{RESET}")

def critic_says(logger:logging.Logger,preview:str) -> None:
    logger.info(f"{YELLOW}  CRITIC-> {preview[:80]}...{RESET}")





