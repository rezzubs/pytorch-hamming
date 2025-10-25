from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import (
    Literal,
    TypeVar,
    cast,
)

import torch

from pytorch_hamming import Data
from pytorch_hamming.encoding import BitPattern
from pytorch_hamming.systems import (
    CachedDataset,
    CachedModel,
    Dtype,
    System,
)
from pytorch_hamming.utils import unreachable

logger = logging.getLogger(__name__)


@dataclass
class Cli:
    """The result of argument parsing."""

    model: CachedModel
    dataset: CachedDataset
    errors: int | float
    dtype: Dtype
    protection: bool | BitPattern
    device: torch.device


def parse_bit_pattern(text: str) -> Literal["all"] | BitPattern:
    if text == "all":
        return text

    try:
        return BitPattern.parse(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Expected a bit pattern or `all`:\n-> {e}")


def parse_num_faults(text: str) -> int:
    try:
        num_faults = int(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"must be a non-negative integer\n-> {e}"
        ) from e

    if num_faults < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return num_faults


def parse_bit_error_rate(text: str) -> float:
    try:
        bit_error_rate = float(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"must be a float betewen 0 and 1 (inclusive)\n-> {e}"
        ) from e

    if not (0 <= bit_error_rate <= 1):
        raise argparse.ArgumentTypeError("mus be between 0 and 1 inclusive")

    return bit_error_rate


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    model_group = parser.add_argument_group("Model setup")

    _ = model_group.add_argument(
        "-m",
        "--model",
        type=str,
        help="The model to experiment on. Required.",
        choices=["resnet20", "vgg11"],
        required=True,
    )

    _ = model_group.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100"],
        help="The dataset to use for evaluation. Required.",
        required=True,
    )

    _ = model_group.add_argument(
        "--dtype",
        "--data-type",
        type=str,
        choices=["f32", "float32", "f16", "float16"],
        default="f32",
        help="Which data type to use for model parameters. Default is f32.",
        required=False,
    )

    _ = model_group.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="The batch size to use for the dataloader. Default is 1000.",
        required=False,
    )

    fi_group = parser.add_argument_group(
        "Fault injection settings",
        "Use --bit-error-rate or --num-faults to enable fault injection.",
    )

    fi_group_mut = fi_group.add_mutually_exclusive_group()

    # TODO: custom parser
    _ = fi_group_mut.add_argument(
        "--bit-error-rate",
        type=parse_bit_error_rate,
        help="Bit error rate to use for fault injection.",
        required=False,
    )
    _ = fi_group_mut.add_argument(
        "--num-faults",
        type=parse_num_faults,
        help="Number of faults to inject. All bit flips are going to be unique.",
        required=False,
    )

    protect_group = parser.add_argument_group(
        "Encoding settings",
        "These settings are used to protect the model during fault injection. \
        Use --bits/--protected to enable encoding.",
    )

    pattern_group = protect_group.add_mutually_exclusive_group()

    _ = pattern_group.add_argument(
        "--bits",
        "--bit-pattern",
        type=parse_bit_pattern,
        help='Comma separated list of bit indices to protect or "all". \
        The bit indices cannot exceed the number of bits in the data type.',
        required=False,
    )

    _ = pattern_group.add_argument(
        "--protected",
        action="store_true",
        help="Alias for --bits=all",
        required=False,
    )

    _ = protect_group.add_argument(
        "--memory-line",
        type=int,
        default=64,
        help="How many bits are in a memory line. Must be a multiple of dtype size. \
        Default is 64 bits.",
        required=False,
    )

    # TODO: custom parser
    _ = protect_group.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="How many memory lines to encode as one chunk. \
        Doubling the block size has the same effect as doubling the memory line size. \
        Default is 1",
        required=False,
    )

    other_group = parser.add_argument_group("Other settings")

    _ = other_group.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cpu"),
        help="A pytorch device string, for example `cuda:0`. Default is `cpu`",
        required=False,
    )

    return parser


T = TypeVar("T")


def get_arg_typed(args: argparse.Namespace, name: str, expected: type[T]) -> T:
    arg = getattr(args, name)  # pyright: ignore[reportAny]
    assert isinstance(arg, expected)
    return arg


def get_arg_typed_opt(
    args: argparse.Namespace, name: str, expected: type[T]
) -> T | None:
    arg = getattr(args, name)  # pyright: ignore[reportAny]
    assert isinstance(arg, expected | None)
    return arg


def parse_cli() -> Cli:
    args = create_parser().parse_args()

    match get_arg_typed(args, "model", str):
        case "resnet20":
            model = CachedModel.ResNet20
        case "vgg11":
            model = CachedModel.VGG11
        case other:
            unreachable(other)

    match get_arg_typed(args, "dataset", str):
        case "cifar10":
            dataset = CachedDataset.CIFAR10
        case "cifar100":
            dataset = CachedDataset.CIFAR100
        case other:
            unreachable(other)

    match (
        get_arg_typed_opt(args, "num_faults", int),
        get_arg_typed_opt(args, "bit_error_rate", float),
    ):
        case (None, None):
            errors = 0
        case (num_faults, None):
            errors = num_faults
        case (None, bit_error_rate):
            errors = bit_error_rate
        case other:
            unreachable(other)

    match get_arg_typed(args, "dtype", str):
        case "float32" | "f32":
            dtype = Dtype.Float32
        case "float16" | "f16":
            dtype = Dtype.Float16
        case other:
            unreachable(other)

    bits = cast(BitPattern | Literal["all"] | None, args.bits)
    assert isinstance(bits, BitPattern) or isinstance(bits, str) or bits is None

    match (bits, get_arg_typed(args, "protected", bool)):
        case ("all", _) | (_, True):
            protection = True
        case (None, False):
            protection = False
        case (BitPattern(), False):
            protection = bits

    device = get_arg_typed(args, "device", torch.device)

    return Cli(model, dataset, errors, dtype, protection, device)


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


def main():
    logging.basicConfig(level=get_log_level())

    cli = parse_cli()

    system = System(cli.dataset, cli.model, cli.dtype, cli.device)

    total_num_bits = system.system_total_num_bits()

    match cli.errors:
        case int():
            num_faults = total_num_bits
        case float(val):
            num_faults = int(round(total_num_bits * val))

    data = Data.load_or_create(
        "temp.json",
        faults_count=num_faults,
        bits_count=total_num_bits,
        metadata=system.system_metadata(),
    )

    logger.debug(f"Proceeding with data: {data}")

    _ = data.record_entry(system)
