from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import hamming_core
import torch
from typing_extensions import override

from pytorch_hamming.dtype import DnnDtype
from pytorch_hamming.tensor_ops import tensor_list_dtype

logger = logging.getLogger(__name__)


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

    @override
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

    @override
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

    @override
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
        bits: set[int] = set()

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


@dataclass
class BitPatternEncoding:
    _encoded_data: hamming_core.BitPatternEncoding
    _decoded_tensors: list[torch.Tensor] | None

    @classmethod
    def encode_tensor_list(
        cls,
        ts: list[torch.Tensor],
        pattern: BitPattern,
        pattern_length: int,
        bits_per_chunk: int,
    ) -> BitPatternEncoding:
        dtype = tensor_list_dtype(ts)
        if dtype is None:
            raise ValueError("Cannot encode an empty buffer")

        flattened = [t.flatten() for t in ts]

        match DnnDtype.from_torch(dtype):
            case DnnDtype.Float32:
                with torch.no_grad():
                    rust_input = [t.numpy(force=True) for t in flattened]
                    data = hamming_core.encode_bit_pattern_f32(
                        rust_input,
                        pattern.bits,
                        pattern_length,
                        bits_per_chunk,
                    )
            case DnnDtype.Float16:
                with torch.no_grad():
                    rust_input = [
                        t.view(torch.uint16).numpy(force=True) for t in flattened
                    ]
                    data = hamming_core.encode_bit_pattern_u16(
                        rust_input,
                        pattern.bits,
                        pattern_length,
                        bits_per_chunk,
                    )

        flattened_copy = [t.clone() for t in flattened]
        return cls(data, flattened_copy)

    def decode_tensor_list(
        self, output_buffer: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Decode into the output buffer.

        `output_buffer` should have the same structure as the original data.

        Returns a reference to the same output buffer.
        """
        if self._decoded_tensors is not None:
            logger.debug("Using existing value for decoded tensors")
            for original, decoded in zip(
                output_buffer, self._decoded_tensors, strict=True
            ):
                with torch.no_grad():
                    # Discard because it's updated inplace.
                    _ = original.flatten().copy_(decoded)
            return output_buffer

        logger.debug("Recomputing decoded tensors")

        dtype = tensor_list_dtype(output_buffer)
        if dtype is None:
            raise ValueError("Cannot decode into an empty buffer")

        element_counts = [t.numel() for t in output_buffer]
        match DnnDtype.from_torch(dtype):
            case DnnDtype.Float32:
                decoded, ded_results = self._encoded_data.decode_bit_pattern_f32(
                    element_counts
                )
                # HACK: There's nothing we can do about this warning without an upstream fix.
                torch_decoded = [
                    torch.from_numpy(t)  # pyright: ignore[reportUnknownMemberType]
                    for t in decoded
                ]

            case DnnDtype.Float16:
                decoded, ded_results = self._encoded_data.decode_bit_pattern_u16(
                    element_counts
                )
                # HACK: There's nothing we can do about this warning without an upstream fix.
                torch_decoded = [
                    torch.from_numpy(t).view(torch.float16)  # pyright: ignore[reportUnknownMemberType]
                    for t in decoded
                ]

        self._decoded_tensors = torch_decoded
        for original, decoded in zip(output_buffer, torch_decoded, strict=True):
            with torch.no_grad():
                # Discard because it's updated inplace.
                _ = original.flatten().copy_(decoded)

        # TODO: we discard the double error detection results for now but may
        # want to do something with them in the future.
        _ = ded_results

        return output_buffer

    def flip_n_bits(self, n: int):
        logger.debug("Invalidating decoded tensor cached due to fault injection.")
        self._decoded_tensors = None
        self._encoded_data.flip_n_bits(n)

    def clone(self) -> BitPatternEncoding:
        copied_tensors = (
            [t.clone() for t in self._decoded_tensors]
            if self._decoded_tensors is not None
            else None
        )
        return BitPatternEncoding(self._encoded_data.clone(), copied_tensors)

    def bits_count(self) -> int:
        return self._encoded_data.bits_count()
