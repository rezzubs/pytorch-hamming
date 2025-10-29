import logging
import os


def get_log_level():
    try:
        level = os.environ["LOG_LEVEL"]
    except KeyError:
        return logging.INFO

    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    try:
        return levels[level.lower()]
    except KeyError:
        print(
            f"invalid log level `{level}`, expected one of: {', '.join(levels.keys())}"
        )
        exit(1)


def setup_logging():
    logging.basicConfig(level=get_log_level())
