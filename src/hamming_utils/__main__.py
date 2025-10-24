import logging
import os

from hamming_utils import Data
from hamming_utils.cli import parse_cli
from hamming_utils.systems import System


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
        print(f"invalid log level {level}, expected one of: {levels}")
        exit(1)


def main():
    logging.basicConfig(level=get_log_level())

    cli = parse_cli()

    system = System(cli.dataset, cli.model, cli.dtype, cli.device)

    data = Data.load_or_create(
        "temp.json",
        num_faults=cli.errors,
        num_bits=system.system_total_num_bits(),
        metadata=system.system_metadata(),
    )

    print(cli)


if __name__ == "__main__":
    _ = main()
