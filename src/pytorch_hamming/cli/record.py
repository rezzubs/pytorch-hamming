from __future__ import annotations

import enum
import logging
from typing import (
    Annotated,
)

import torch
import typer

from pytorch_hamming import (
    Data,
    DnnDtype,
)
from pytorch_hamming.cli.utils import setup_logging
from pytorch_hamming.encoding import (
    BitPattern,
    EncodedSystem,
    EncodingFormatFull,
)
from pytorch_hamming.systems import (
    CachedDataset,
    CachedModel,
    System,
)

logger = logging.getLogger(__name__)

app = typer.Typer()


class DtypeChoices(enum.StrEnum):
    F16 = "f16"
    Float16 = "float16"
    F32 = "f32"
    Float32 = "float32"

    def to_dtype(self) -> DnnDtype:
        match self.value:
            case "f16" | "float16":
                return DnnDtype.Float16
            case "f32" | "float32":
                return DnnDtype.Float32


@app.command()
def record(
    model: Annotated[
        CachedModel,
        typer.Option(
            "--model",
            "-m",
            prompt=True,
            help="The model to experiment on.",
            rich_help_panel="Model setup",
        ),
    ],
    dataset: Annotated[
        CachedDataset,
        typer.Option(
            "--dataset",
            "-d",
            prompt=True,
            help="The dataset to use for evaluation.",
            rich_help_panel="Model setup",
        ),
    ],
    dtype: Annotated[
        DtypeChoices,
        typer.Option(
            help="The data type to use for the model.",
            rich_help_panel="Model setup",
        ),
    ] = DtypeChoices.F32,
    batch_size: Annotated[
        int,
        typer.Option(
            help="The batch size to use for the data loader.",
            rich_help_panel="Model setup",
        ),
    ] = 1000,
    bit_error_rate: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=1.0,
            help="The bit error rate to use for fault injection. Incompatible with --faults-count",
            rich_help_panel="Fault injection settings",
        ),
    ] = None,
    faults_count: Annotated[
        int | None,
        typer.Option(
            min=0,
            help="How many faults to inject into the model. Incompatible with --bit_error_rate",
            rich_help_panel="Fault injection settings",
        ),
    ] = None,
    protected: Annotated[
        bool,
        typer.Option(
            "--encoded",
            "--protected",
            help="Encode all bits",
            rich_help_panel="Encoding settings",
        ),
    ] = False,
    bit_pattern: Annotated[
        BitPattern | None,
        typer.Option(
            "--bit-pattern",
            "--bits",
            parser=BitPattern.parse,
            help="`_` separated bit indices or ranges of indices \
For example, 0-4_14_18-19. \
The bit indices cannot exceed the number of bits in the data type.",
            rich_help_panel="Encoding settings",
        ),
    ] = None,
    bits_per_chunk: Annotated[
        int,
        typer.Option(
            "--chunk-size",
            "--bits-per_chunk",
            help="How many bits to encode as a chunk. \
For example if we want to have 1 ecc per 2 float32 values (with all bits protected), \
a chunk size of 64 should be used.",
            rich_help_panel="Encoding settings",
        ),
    ] = 64,
    device: Annotated[  # pyright: ignore[reportRedeclaration]
        str,
        typer.Option(
            help="A pytorch device string, for example `cuda:0`.",
        ),
    ] = "cpu",
):
    """Record data entries for a model and dataset."""
    device: torch.device = torch.device(device)

    system = System(dataset=dataset, model=model, dtype=dtype.to_dtype(), device=device)

    match (protected, bit_pattern):
        case (_, BitPattern()):
            raise NotImplementedError
        case (True, None):
            system = EncodedSystem(system, EncodingFormatFull(bits_per_chunk))
        case _:
            pass

    total_num_bits = system.system_total_num_bits()

    match (faults_count, bit_error_rate):
        case (None, None):
            faults_count = 0
        case (faults_count, None):
            faults_count = faults_count
        case (None, bit_error_rate):
            faults_count = int(round(total_num_bits * bit_error_rate))
        case _:
            raise typer.Abort("Choose one of --bit_error_rate and --faults_count")

    data = Data.load_or_create(
        "temp.json",
        faults_count=faults_count,
        bits_count=total_num_bits,
        metadata=system.system_metadata(),
    )

    logger.debug(f"Proceeding with data: {data}")

    # NOTE: The match is otherwise redundant but we're using it to satisfy the
    # type checker. An alternative would be using `Any` to erase the `T` in
    # `BaseSystem[T]` but this causes various "partially unknown warnings".
    match system:
        case System():
            _ = data.record_entry(system)
        case EncodedSystem():
            _ = data.record_entry(system)


if __name__ == "__main__":
    setup_logging()
    app()
