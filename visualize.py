from dataclasses import dataclass
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np


from pytorch_ecc._data import Data
from pytorch_ecc._hamming import HammingStats


@dataclass
class Statistic:
    name: str
    x_axis: str
    y_axis: str
    marker: float
    entries: list[float]

    def average(self) -> float:
        return float(np.average(self.entries))

    def std(self) -> float:
        return float(np.std(self.entries))

    def min(self) -> float:
        return float(np.min(self.entries))

    def max(self) -> float:
        return float(np.max(self.entries))


def draw_statistics(stats: list[Statistic], ax: Axes) -> None:
    for s in stats:
        print(s.marker, len(s.entries))
    markers = [s.marker for s in stats]
    average = [s.average() for s in stats]
    std = [s.std() for s in stats]
    min = [s.min() for s in stats]
    max = [s.max() for s in stats]

    ax.set_xlabel(stats[0].x_axis)
    ax.set_ylabel(stats[0].y_axis)
    ax.set_title(stats[0].name)

    ax.set_xscale("log", base=10)
    ax.plot(markers, average, color="C0")
    ax.errorbar(markers, average, yerr=std, capsize=5, color="C0")
    ax.scatter(markers, min, s=4, color="C0")
    ax.scatter(markers, max, s=4, color="C0")


def draw_data_stats(data: Data):
    protected: dict[float, list[HammingStats]] = dict()
    unprotected: dict[float, list[HammingStats]] = dict()

    for e in data.entries:
        target = protected if e.was_protected else unprotected
        target_ber = target.get(e.bit_error_rate(), [])
        target_ber.append(e)
        target[e.bit_error_rate()] = target_ber

    protected_base = [(k, v) for k, v in protected.items()]
    protected_base.sort(key=lambda x: x[0])

    unprotected_base = [(k, v) for k, v in unprotected.items()]
    unprotected_base.sort(key=lambda x: x[0])

    protected_accuracy = [
        Statistic(
            "Accuracy of protected",
            "BER",
            "%",
            ber,
            [e.get_accuracy() for e in entries],
        )
        for ber, entries in protected_base
    ]

    protected_container_2_or_more_faults = [
        Statistic(
            "Cases of 2 or more faults in protected container",
            "BER",
            "count",
            ber,
            [
                len([v for v in e.faulty_containers().values() if len(v) >= 2])
                for e in entries
            ],
        )
        for ber, entries in protected_base
    ]

    output_bit_error_rate = [
        Statistic(
            "BER of decoded parameters",
            "BER",
            "Output BER",
            ber,
            [e.output_bit_error_rate() for e in entries],
        )
        for ber, entries in protected_base
    ]

    protection_rate = [
        Statistic(
            "Rate of sucessful protections",
            "BER",
            "rate",
            ber,
            [e.protection_rate() for e in entries],
        )
        for ber, entries in protected_base
    ]

    unprotected_accuracy = [
        Statistic(
            "Accuracy of unprotected",
            "BER",
            "%",
            ber,
            [e.get_accuracy() for e in entries],
        )
        for ber, entries in unprotected_base
    ]

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, ncols=1, figsize=(6, 18))

    draw_statistics(unprotected_accuracy, ax0)
    draw_statistics(protected_accuracy, ax1)
    draw_statistics(output_bit_error_rate, ax2)
    draw_statistics(protected_container_2_or_more_faults, ax3)
    draw_statistics(protection_rate, ax4)

    plt.tight_layout()
    plt.savefig("result.png", dpi=400)


if __name__ == "__main__":
    data = Data.load("../")
    draw_data_stats(data)
