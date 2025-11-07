from pathlib import Path
from typing import Annotated
from pytorch_hamming import Data
import typer
import matplotlib.patches as patches
from matplotlib import pyplot as plt

app = typer.Typer()


def plot(file: Path, stability_threshold: float, stable_within: int):
    data = Data.load(file)

    fig, ax = plt.subplots()  # pyright: ignore[reportUnknownMemberType]

    _ = ax.set_title(  # pyright: ignore[reportUnknownMemberType]
        "\n".join(
            [f"{k}={v}" for k, v in data.metadata.items()]
            + [f"BER={data.faults_count / data.bits_count:.2e}"]
        )
    )

    means = data.means()

    _ = ax.plot(means)  # pyright: ignore[reportUnknownMemberType]

    result = data.mean_drift(stable_within)
    if result is None:
        plt.show()  # pyright: ignore[reportUnknownMemberType]
        return

    drift_min, drift_max = result

    num_entries = len(data.entries)

    box_start_x = num_entries - stable_within

    drift = drift_max - drift_min

    edge_color = "green" if drift <= stability_threshold else "red"

    rect = patches.Rectangle(
        (box_start_x, drift_min),
        stable_within,
        drift,
        facecolor="none",
        edgecolor=edge_color,
    )

    _ = ax.add_patch(rect)
    _ = ax.text(box_start_x, drift_max, f"drift {drift:.2}%")  # pyright: ignore[reportUnknownMemberType]

    _ = ax.set_xlabel("number of runs")  # pyright: ignore[reportUnknownMemberType]
    _ = ax.set_ylabel("accuracy [%]")  # pyright: ignore[reportUnknownMemberType]

    fig.tight_layout()
    plt.show()  # pyright: ignore[reportUnknownMemberType]


def recurse(path: Path, stability_threshold: float, stable_within: int):
    if path.is_dir():
        for subpath in path.iterdir():
            recurse(subpath, stability_threshold, stable_within)
    else:
        plot(path, stability_threshold, stable_within)


@app.command()
def draw_mean(
    data_path: Annotated[
        Path,
        typer.Argument(
            help="the file where data was recorded",
        ),
    ],
    stability_threshold: Annotated[
        float,
        typer.Option(
            min=0,
            max=100,
            help="The percentage of mean drift that's considered stable",
        ),
    ] = 1,
    stable_within: Annotated[
        int,
        typer.Option(
            min=1,
        ),
    ] = 100,
):
    recurse(data_path, stability_threshold, stable_within)
