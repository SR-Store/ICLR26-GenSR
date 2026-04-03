

import logging
import time
import os
from datetime import timedelta
import shutil
from pathlib import Path


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(filepath, rank: int):
    log_formatter = LogFormatter()

    file_handler = None
    if filepath is not None:
        if rank > 0:
            filepath = f"{filepath}-{rank}"

        file_path = Path(filepath)
        log_dir   = file_path.parent

        if rank == 0 and log_dir.exists():
            shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(filepath, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if file_handler:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger

