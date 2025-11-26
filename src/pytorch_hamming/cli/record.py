from __future__ import annotations

import enum
import logging
from pathlib import Path
from typing import (
    Annotated,
    Any,
    cast,
)

import torch
import typer

from pytorch_hamming.cifar_models.dataset import CachedDataset as Cifar
from pytorch_hamming.cifar_models.model import CachedModel as CifarModel
from pytorch_hamming.cifar_models.system import System as CifarSystem
from pytorch_hamming.cli.utils import setup_logging
from pytorch_hamming.data import (
    Autosave,
    Data,
)
from pytorch_hamming.dtype import DnnDtype
from pytorch_hamming.encoding.bit_pattern import BitPattern
from pytorch_hamming.encoding.system import (
    EncodedSystem,
    EncodingFormatBitPattern,
    EncodingFormatFull,
    EncodingFormatMsb,
)
from pytorch_hamming.system import BaseSystem

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
        CifarModel,
        typer.Option(
            "--model",
            "-m",
            prompt=True,
            help="The model to experiment on.",
            rich_help_panel="Model setup",
        ),
    ],
    dataset: Annotated[
        Cifar.Kind,
        typer.Option(
            "--dataset",
            "-d",
            prompt=True,
            help="The dataset to use for evaluation.",
            rich_help_panel="Model setup",
        ),
    ],
    dataset_cache: Annotated[
        Path | None,
        typer.Option(
            "--dataset-cache",
            help="The path to use for caching the dataset. `./dataset-cache` by default.",
            rich_help_panel="Model setup",
        ),
    ] = None,
    dtype: Annotated[  # pyright: ignore[reportRedeclaration]
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
    duplicate_msb: Annotated[
        bool,
        typer.Option(
            help="Duplicate the most significant bit inside two lowest ones.",
            rich_help_panel="Encoding settings",
        ),
    ] = False,
    device: Annotated[  # pyright: ignore[reportRedeclaration]
        str,
        typer.Option(
            help="A pytorch device string, for example `cuda:0`.",
        ),
    ] = "cpu",
    runs: Annotated[
        int | None,
        typer.Option(
            min=2,
            help="How many runs to perform.",
            rich_help_panel="Recording settings. 1 run is done by default.",
        ),
    ] = None,
    until_stable: Annotated[
        int | None,
        typer.Option(
            min=2,
            help="run until the accuracy mean within this many runs has not fluctuated over --stability-threshold %. \
--runs can be used to signal the minimum number of runs",
            rich_help_panel="Recording settings",
        ),
    ] = None,
    stability_threshold: Annotated[
        float,
        typer.Option(
            min=0,
            max=100,
            help="run until the accuracy mean within this many runs has not fluctuated over --stability-threshold %",
            rich_help_panel="Recording settings",
        ),
    ] = 1,
    output_path: Annotated[
        Path | None,
        typer.Option(
            help="The path of the file to save the results into. \
If the file doesn't exist then it will be created. \
If the path is a directory (existing) then the file will be called data.json",
            rich_help_panel="Recording settings",
        ),
    ] = None,
    metadata_name: Annotated[
        bool,
        typer.Option(
            help="Generate a file name based on the configuration metadata. Must be used together with --output-path.",
            rich_help_panel="Recording settings",
        ),
    ] = False,
    autosave: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="How often to save the data while recording, interval - number of runs. \
The default is to only save at the very end",
            rich_help_panel="Recording settings",
        ),
    ] = None,
    summary: Annotated[
        bool,
        typer.Option(
            help="Print a summary at the end of each evaluation",
            rich_help_panel="Recording settings",
        ),
    ] = False,
):
    """Record data entries for a model and dataset."""
    device: torch.device = torch.device(device)

    dtype: DnnDtype = dtype.to_dtype()

    if dataset_cache is None:
        dataset_cache = Path("./dataset-cache")
    dataset_cache = dataset_cache.expanduser()

    if not dataset_cache.exists():
        logger.info(f"Creating a new cache directory at `{dataset_cache}`")
        dataset_cache.mkdir(parents=True)

    if not dataset_cache.is_dir():
        print(f"Dataset cache ({dataset_cache}) must be a directory")
        raise typer.Exit()

    system = cast(
        BaseSystem[Any],
        CifarSystem(
            dataset=Cifar(dataset, dataset_cache),
            model=model,
            dtype=dtype,
            device=device,
            batch_size=batch_size,
            dataset_cache=dataset_cache,
        ),
    )

    if duplicate_msb:
        system = cast(BaseSystem[Any], EncodedSystem(system, EncodingFormatMsb()))

    match (protected, bit_pattern):
        case (_, BitPattern()):
            system = EncodedSystem(
                system,
                EncodingFormatBitPattern(
                    pattern=bit_pattern,
                    pattern_length=dtype.bits_count(),
                    bits_per_chunk=bits_per_chunk,
                ),
            )
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
            print("Choose one of --bit_error_rate and --faults_count")
            raise typer.Exit()

    if output_path is not None:
        output_path = output_path.expanduser()

    data = Data.load_or_create(
        output_path,
        faults_count=faults_count,
        bits_count=total_num_bits,
        metadata=system.system_metadata(),
        metadata_name=metadata_name,
    )

    logger.debug(f"Proceeding with metadata: {data.metadata}")

    match (runs, until_stable):
        case (None, None):
            _ = data.record_entry(
                cast(BaseSystem[Any], system),
                summary=summary,
            )
        case (_, None):
            if autosave is not None and output_path is not None:
                save_config = Autosave(autosave, output_path, metadata_name)
            else:
                save_config = None

            data.record_entries(
                cast(BaseSystem[Any], system),
                runs,
                summary=summary,
                autosave=save_config,
            )
        case _:
            if autosave is not None and output_path is not None:
                save_config = Autosave(autosave, output_path, metadata_name)
            else:
                save_config = None

            _ = data.record_until_stable(
                cast(BaseSystem[Any], system),
                threshold=stability_threshold,
                stable_within=until_stable,
                min_runs=runs,
                autosave=save_config,
            )

    if output_path:
        data.save(output_path, metadata_name)


if __name__ == "__main__":
    setup_logging()
    app()
