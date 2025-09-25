import matplotlib.pyplot as plt

from hamming_utils import Data
from sys import argv
from pathlib import Path
from collections.abc import Iterable


def main() -> None:
    data_path = Path(argv[1])

    paths = []
    if data_path.is_dir():
        for child in data_path.iterdir():
            if child.is_file():
                paths.append(child)
    else:
        paths.append(data_path)

    paths.sort()

    datas = [Data.load(path, None) for path in paths]

    fig, axes = plt.subplots(len(datas), figsize=(6, 3 * len(datas)))

    if isinstance(axes, Iterable):
        for data, ax in zip(datas, axes):
            data.plot_accuracy(ax)
    else:
        datas[0].plot_accuracy(axes)

    plt.tight_layout()

    try:
        save_path = argv[2]
        print(f"saving to {save_path}")
        fig.savefig(save_path)
    except IndexError:
        plt.show()


if __name__ == "__main__":
    main()
