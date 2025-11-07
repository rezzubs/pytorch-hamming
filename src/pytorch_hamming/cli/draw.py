import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer

from pytorch_hamming import Data

logger = logging.getLogger(__name__)

app = typer.Typer()


def path_sequence_entries(paths: Iterable[Path]) -> list[Data.Entry]:
    entries: list[Data.Entry] = []

    for path in paths:
        path = path.expanduser()

        if path.is_dir():
            entries.extend(path_sequence_entries(path.iterdir()))
            continue

        try:
            data = Data.load(path)
        except Exception as e:
            logger.warning(f"Failed to load data from path `{path}` - skipping\n-> {e}")
            continue

        entries.extend(data.entries)

    return entries


@app.command()
def scatter(
    datasets: Annotated[
        list[Path],
        typer.Argument(
            help="All the paths that contain data. These paths will be searched recursively."
        ),
    ],
    max_accuracy: Annotated[
        float,
        typer.Option(
            max=100,
            min=0,
            help="Filter the values above this accuracy threshold.",
        ),
    ] = 100,
    min_accuracy: Annotated[
        float,
        typer.Option(
            max=100,
            min=0,
            help="Filter the values above this accuracy threshold.",
        ),
    ] = 0,
    use_3d: Annotated[
        bool,
        typer.Option(
            "--3d",
            help="Draw with a 3d projection",
        ),
    ] = False,
    max_total_faults: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Skip the entries where the total number of faults exceeds this value.",
        ),
    ] = None,
    skip_multi_bit_faults: Annotated[
        bool,
        typer.Option(
            help="Skip the cases where there were multiple faults per parameter.",
        ),
    ] = False,
    inverse_drawing_order: Annotated[
        bool,
        typer.Option(
            help="Draw the cases with the lowest number of faults per bit last",
        ),
    ] = False,
):
    """Draw a scatter plot for all the entries in the given data.

    The z-axis corresponds to the number of faults in each bit for a given accuracy.
    """
    entries = path_sequence_entries(datasets)

    xs: list[float] = []
    ys: list[int] = []
    zs: list[int] = []

    total_xs: list[float] = []
    total_ys: list[int] = []

    for entry in entries:
        if entry.accuracy > max_accuracy or entry.accuracy < min_accuracy:
            continue

        fault_map = entry.faults_per_bit_index(skip_multi_bit_faults)
        total_faults = sum(fault_map.values())

        if max_total_faults is not None and total_faults > max_total_faults:
            continue

        total_xs.append(entry.accuracy)
        total_ys.append(total_faults)

        for bit_index, faults_count in fault_map.items():
            if faults_count == 0:
                continue
            xs.append(entry.accuracy)
            ys.append(bit_index)
            zs.append(faults_count)

    xs_ = np.array(xs)
    ys_ = np.array(ys)
    zs_ = np.array(zs)

    if inverse_drawing_order:
        order = np.argsort(-zs_)
    else:
        order = np.argsort(zs_)

    total_xs_ = np.array(total_xs)
    total_ys_ = np.array(total_ys)

    total_order = np.argsort(total_ys_)

    plt.style.use("dark_background")

    fig = plt.figure()  # pyright: ignore[reportUnknownMemberType]

    if use_3d:
        ax = fig.subplots(subplot_kw=dict(projection="3d"))

        scatter_ = ax.scatter(  # pyright: ignore[reportUnknownMemberType]
            xs_[order], ys_[order], zs_[order], c=zs_[order], cmap="plasma"
        )
        _ = ax.set_xlabel("Accuracy [%]")  # pyright: ignore[reportUnknownMemberType]
        _ = ax.set_ylabel("Bit index")  # pyright: ignore[reportUnknownMemberType]
        ax.set_zlabel("Number of faults")  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

        _ = plt.colorbar(scatter_, ax=ax)  # pyright: ignore[reportUnknownMemberType]
    else:
        ax = fig.subplot_mosaic(
            [["top", "cbar"], ["bot", "cbar"]],
            width_ratios=[0.95, 0.05],
            height_ratios=[1 / 3, 2 / 3],
        )
        ax["top"].sharex(ax["bot"])

        scatter_ = ax["bot"].scatter(  # pyright: ignore[reportUnknownMemberType]
            xs_[order], ys_[order], c=zs_[order], cmap="plasma"
        )
        _ = ax["bot"].set_xlabel("Accuracy [%]")  # pyright: ignore[reportUnknownMemberType]
        _ = ax["bot"].set_ylabel("Bit index")  # pyright: ignore[reportUnknownMemberType]

        _ = ax["top"].scatter(total_xs_[total_order], total_ys_[total_order])  # pyright: ignore[reportUnknownMemberType]
        _ = ax["top"].set_ylabel("Total faults")  # pyright: ignore[reportUnknownMemberType]
        ax["top"].set_yscale("log")  # pyright: ignore[reportUnknownMemberType]

        _ = fig.colorbar(scatter_, ax["cbar"])  # pyright: ignore[reportUnknownMemberType]
        _ = ax["cbar"].set_ylabel("Number of faults per bit index")  # pyright: ignore[reportUnknownMemberType]

    fig.tight_layout()
    plt.show()  # pyright: ignore[reportUnknownMemberType]
