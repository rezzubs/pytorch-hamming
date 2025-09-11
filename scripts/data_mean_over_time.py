import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from hamming_utils import Data


def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("path", metavar="DATA_PATH", type=str)

    args = parser.parse_args()

    data = Data.load(args.path).partition()

    _, axes = plt.subplots(len(data), figsize=(6, 3 * len(data)))

    max_len = 0
    for i, (key, val) in enumerate(data.items()):
        val = val.get("protected")
        if val is None:
            continue

        max_len = max(max_len, len(val))

        means = []
        for j in range(len(val)):
            means.append(np.mean(val[:j]))

        ax = axes[i]
        assert isinstance(ax, Axes)
        ax.set_title(str(key))
        ax.plot(means)

    for ax in axes:
        ax.set_xlim(0, max_len)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
