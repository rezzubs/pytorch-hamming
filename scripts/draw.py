from dataclasses import dataclass
import matplotlib.pyplot as plt

from hamming_utils import Data
from sys import argv
from pathlib import Path
from collections.abc import Iterable


@dataclass
class Group:
    items: list[Data]
    baseline: int | None


def partition(datas: list[Data]):
    groups: dict[str, Group] = dict()
    for data in datas:
        dtype = data.meta.dtype
        group = groups.get(dtype, Group([], None))
        if data.meta.buffer_size is None:
            group.baseline = data.entries[0].total_bits

        group.items.append(data)
        groups[dtype] = group

    for group in groups.values():
        group.items.sort(key=lambda x: x.meta.buffer_size or 0)

    groups_list = list(groups.items())
    groups_list.sort(key=lambda x: x[0])

    return groups_list


def main() -> None:
    data_path = Path(argv[1])

    paths = []
    if data_path.is_dir():
        for child in data_path.iterdir():
            if child.is_file():
                paths.append(child)
    else:
        paths.append(data_path)

    datas = [Data.load(path, None) for path in paths]

    groups = partition(datas)

    width = 0
    height = 0
    for _, group in groups:
        height += 1
        width = max(width, len(group.items))

    fig, axes = plt.subplots(height, width, figsize=(6 * width, 3 * height))

    for y, row in enumerate(axes):
        _, group = groups[y]
        for x, ax in enumerate(row):
            try:
                item = group.items[x]
            except IndexError:
                continue
            item.plot_accuracy(ax, group.baseline)

    plt.tight_layout()

    try:
        save_path = argv[2]
        print(f"saving to {save_path}")
        fig.savefig(save_path)
    except IndexError:
        plt.show()


if __name__ == "__main__":
    main()
