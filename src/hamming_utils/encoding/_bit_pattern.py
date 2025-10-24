from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class RangeInclusive:
    start: int
    end: int

    def __init__(self, start: int | str, end: int | str) -> None:
        try:
            if (start := int(start)) < 0:
                raise ValueError("Range values must be non-negative")
        except ValueError as e:
            raise ValueError(f"Invalid range start value `{start}`\n-> {e}") from e

        try:
            if (end := int(end)) < 0:
                raise ValueError("Range values must be non-negative")
        except ValueError as e:
            raise ValueError(f"Invalid range end value `{end}`\n-> {e}") from e

        self.start = start
        self.end = end

    def __repr__(self) -> str:
        if self.start == self.end:
            return str(self.start)
        elif self.end > self.start:
            return f"{self.start}-{self.end}"
        else:
            raise RuntimeError(f"Invaild range {self.start}-{self.end}")

    def __iter__(self) -> Iterator[int]:
        return range(self.start, self.end + 1).__iter__()


def parse_range_or_int(text: str) -> RangeInclusive | int:
    split = text.split("-")
    if len(split) == 1:
        try:
            return int(split[0])
        except ValueError as e:
            raise ValueError(
                f"Expected an integer or a `-` separated range, got `{text}`\n-> {e}"
            ) from e

    try:
        [start, end] = split
        return RangeInclusive(start, end)
    except ValueError as e:
        raise ValueError(
            f"Invalid range `{text}`, expected two integers separated by a `-`\n-> {e}"
        ) from e


@dataclass
class Ranges:
    ranges: list[RangeInclusive]

    def __repr__(self) -> str:
        return "_".join([r.__repr__() for r in self.ranges])

    def start_new(self, start: int) -> None:
        self.ranges.append(RangeInclusive(start, start))

    def extend_last(self) -> None:
        self.ranges[-1].end += 1


@dataclass
class BitPattern:
    """A pattern of bits to protect.

    The length is inferred from usage.
    """

    bits: set[int]

    def num_bits(self) -> int:
        return len(self.bits)

    def __repr__(self) -> str:
        if self.num_bits() == 0:
            return "empty_bit_pattern"

        bits = list(self.bits)
        bits.sort()

        ranges = Ranges([])
        previous = None
        for bit in bits:
            if previous is None:
                ranges.start_new(bit)
                previous = bit
                continue

            if previous == bit - 1:
                ranges.extend_last()
            else:
                ranges.start_new(bit)

            previous = bit

        return ranges.__repr__()

    @classmethod
    def parse(cls, text: str) -> BitPattern:
        bits = set()

        try:
            for part in text.split("_"):
                range_or_int = parse_range_or_int(part)

                if isinstance(range_or_int, int):
                    bits.add(range_or_int)
                    continue
                assert isinstance(range_or_int, RangeInclusive)

                for i in range_or_int:
                    bits.add(i)
        except ValueError as e:
            raise ValueError(
                f"Invalid bit pattern `{text}`, expected a `_` separated list of integers or ranges\n-> {e}"
            ) from e

        return cls(bits)
