import argparse
from dataclasses import dataclass


@dataclass
class Cli:
    model: str
    dataset: str
    faults: float | str


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    model_group = parser.add_argument_group("Model setup")

    model_group.add_argument(
        "-m",
        "--model",
        type=str,
        help="The model to experiment on. Required.",
        choices=["resnet20", "vgg11"],
        required=True,
    )

    model_group.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100"],
        help="The dataset to use for evaluation. Required.",
        required=True,
    )

    model_group.add_argument(
        "--dtype",
        "--data-type",
        type=str,
        choices=["f32", "float32", "f16", "float16"],
        default="f32",
        help="Which data type to use for model parameters. Default is f32.",
        required=False,
    )

    model_group.add_argument(
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
    fi_group_mut.add_argument(
        "--bit-error-rate",
        type=float,
        help="Bit error rate to use for fault injection.",
        required=False,
    )
    fi_group_mut.add_argument(
        "--num-faults",
        type=int,
        default=0,
        help="Number of faults to inject. All bit flips are going to be unique.",
        required=False,
    )

    protect_group = parser.add_argument_group(
        "Encoding settings",
        "These settings are used to protect the model during fault injection. "
        "Use --bits/--protected to enable encoding.",
    )

    pattern_group = protect_group.add_mutually_exclusive_group()

    # TODO: custom parser
    pattern_group.add_argument(
        "--bits",
        "--bit-pattern",
        help='Comma separated list of bit indices to protect or "all". '
        "The bit indices cannot exceed the number of bits in the data type.",
        required=False,
    )

    pattern_group.add_argument(
        "--protected",
        action="store_true",
        help="Alias for --bits=all",
        required=False,
    )

    # TODO: custom parser
    protect_group.add_argument(
        "--memory-line",
        type=int,
        default=64,
        help="How many bits are in a memory line. Must be a multiple of dtype size. "
        "Default is 64 bits.",
        required=False,
    )

    # TODO: custom parser
    protect_group.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="How many memory lines to encode as one chunk."
        " Doubling the block size has the same effect as doubling the memory line size. "
        "Default is 1",
        required=False,
    )

    other_group = parser.add_argument_group("Other settings")

    other_group.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="A pytorch device string, for example `cuda:0`. Default is `cpu`",
        required=False,
    )

    return parser


def parse_args() -> argparse.Namespace:
    return create_parser().parse_args()
