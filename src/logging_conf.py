import logging
from logging.handlers import RotatingFileHandler
from .config import LOGS

def get_logger(name: str = "pfm2", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # evita duplicar handlers

    logger.setLevel(level)

    # Consola
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    # Archivo rotativo
    fh = RotatingFileHandler(LOGS / "pfm2.log", maxBytes=2_000_000,
                             backupCount=3, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    ))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
