import copy
import logging
from dataclasses import dataclass
from typing import Protocol, Self, override

from torch import Tensor

from pytorch_hamming.encoding.bit_pattern import (
    BitPattern,
    BitPatternEncoding,
)
from pytorch_hamming.encoding.full import FullEncoding
from pytorch_hamming.encoding.msb import MsbEncoding
from pytorch_hamming.system import BaseSystem

logger = logging.getLogger(__name__)


class Encoding(Protocol):
    def decode_tensor_list(self, output_buffer: list[Tensor]) -> None:
        """Decode into the given list.

        The list is expected to have the same shape as the original unencoded
        data
        """
        ...

    def clone(self) -> Self:
        """Return a full clone of self.

        Modifying the clone should not modify the original in any way.
        """
        ...

    def flip_n_bits(self, n: int) -> None:
        """Flip a number of bits in the encoded data."""
        ...

    def bits_count(self) -> int:
        """Return the number of bits used for the encoded data."""
        ...


class EncodingFormat(Protocol):
    def encode_system[T](self, system: BaseSystem[T]) -> Encoding:
        """Encode the given system"""
        ...

    def extra_metadata[T](self, metadata: dict[str, str]) -> None: ...


@dataclass
class EncodingFormatFull:
    bits_per_chunk: int

    def encode_system[T](self, system: BaseSystem[T]) -> Encoding:
        data_tensors = system.system_data_tensors(system.system_data())
        return FullEncoding.encode_tensor_list(data_tensors, self.bits_per_chunk)

    def extra_metadata(self, metadata: dict[str, str]) -> None:
        metadata["chunk_size"] = str(self.bits_per_chunk)


@dataclass
class EncodingFormatBitPattern:
    pattern: BitPattern
    pattern_length: int
    bits_per_chunk: int

    def encode_system[T](self, system: BaseSystem[T]) -> Encoding:
        data_tensors = system.system_data_tensors(system.system_data())
        return BitPatternEncoding.encode_tensor_list(
            ts=data_tensors,
            pattern=self.pattern,
            pattern_length=self.pattern_length,
            bits_per_chunk=self.bits_per_chunk,
        )

    def extra_metadata(self, metadata: dict[str, str]) -> None:
        metadata["bit_pattern"] = f"{self.pattern}({self.pattern_length})"
        metadata["chunk_size"] = str(self.bits_per_chunk)


class EncodingFormatMsb:
    def encode_system[T](self, system: BaseSystem[T]) -> Encoding:
        data_tensors = system.system_data_tensors(system.system_data())
        return MsbEncoding.encode_tensor_list(data_tensors)

    def extra_metadata(self, metadata: dict[str, str]) -> None:
        metadata["duplicated_bits"] = "30-to-0-1"


class EncodedSystem[T](BaseSystem[Encoding]):
    def __init__(
        self,
        base: BaseSystem[T],
        format: EncodingFormat,
    ) -> None:
        self.base: BaseSystem[T] = base
        self.encoded_data: Encoding | None = None
        self.format: EncodingFormat = format

    def encode_base(self) -> Encoding:
        logger.debug("Encoding data tensors")
        return self.format.encode_system(self.base)

    def decoded_data(self, data: Encoding) -> T:
        # NOTE: It's fine that we're modifying the original tensors directly,
        # not through a copy, because we're using the original only for the
        # shape and the data will be overwritten anyway

        inner_data = self.base.system_data()
        inner_data_tensors = self.base.system_data_tensors(inner_data)

        _ = data.decode_tensor_list(inner_data_tensors)

        return inner_data

    @override
    def system_data(self) -> Encoding:
        if self.encoded_data is None:
            self.encoded_data = self.encode_base()

        return self.encoded_data

    @override
    def system_accuracy(self, data: Encoding) -> float:
        return self.base.system_accuracy(self.decoded_data(data))

    @override
    def system_data_tensors(self, data: Encoding) -> list[Tensor]:
        print(data.__class__)
        return self.base.system_data_tensors(self.decoded_data(data))

    @override
    def system_inject_n_faults(self, data: Encoding, n: int):
        data.flip_n_bits(n)

    @override
    def system_metadata(self) -> dict[str, str]:
        metadata = copy.deepcopy(self.base.system_metadata())

        self.format.extra_metadata(metadata)

        overhead = (
            self.system_total_num_bits() / self.base.system_total_num_bits() - 1
        ) * 100
        metadata["memory_overhead"] = f"{overhead:.1f}%"
        metadata["protected"] = "true"

        return metadata

    @override
    def system_clone_data(self, data: Encoding) -> Encoding:
        return data.clone()

    @override
    def system_total_num_bits(self) -> int:
        return self.system_data().bits_count()
