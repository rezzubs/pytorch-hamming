import copy
import logging
from dataclasses import dataclass

from pytorch_hamming.tensor_ops import tensor_list_fault_injection
import torch
from typing_extensions import override

from .._system import BaseSystem
from ._full import FullEncoding
from ._bit_pattern import (
    BitPattern,
    BitPatternEncoding,
)

logger = logging.getLogger(__name__)


@dataclass
class EncodingFormatFull:
    bits_per_chunk: int


@dataclass
class EncodingFormatBitPattern:
    pattern: BitPattern
    pattern_length: int
    bits_per_chunk: int


type EncodingFormat = EncodingFormatFull | EncodingFormatBitPattern

type Encoding = FullEncoding | BitPatternEncoding


class EncodedSystem[T](BaseSystem[Encoding]):
    def __init__(
        self,
        base: BaseSystem[T],
        format: EncodingFormat,
    ) -> None:
        self.inner: BaseSystem[T] = base
        self.encoded_data: Encoding | None = None
        self.format: EncodingFormat = format

    def encode_base(self) -> Encoding:
        logger.debug("Encoding data tensors")
        match self.format:
            case EncodingFormatFull(bits_per_chunk):
                data_tensors = self.inner.system_data_tensors(self.inner.system_data())
                return FullEncoding.encode_tensor_list(data_tensors, bits_per_chunk)
            case EncodingFormatBitPattern(pattern, pattern_length, bits_per_chunk):
                data_tensors = self.inner.system_data_tensors(self.inner.system_data())
                return BitPatternEncoding.encode_tensor_list(
                    ts=data_tensors,
                    pattern=pattern,
                    pattern_length=pattern_length,
                    bits_per_chunk=bits_per_chunk,
                )

    def decoded_data(self, data: Encoding) -> T:
        # NOTE: It's fine that we're modifying the original tensors directly,
        # not through a copy, because we're using the original only for the
        # shape and the data will be overwritten anyway

        inner_data = self.inner.system_data()
        inner_data_tensors = self.inner.system_data_tensors(inner_data)

        _ = data.decode_tensor_list(inner_data_tensors)

        return inner_data

    @override
    def system_data(self) -> Encoding:
        if self.encoded_data is None:
            self.encoded_data = self.encode_base()

        return self.encoded_data

    @override
    def system_accuracy(self, data: Encoding) -> float:
        return self.inner.system_accuracy(self.decoded_data(data))

    @override
    def system_data_tensors(self, data: Encoding) -> list[torch.Tensor]:
        return self.inner.system_data_tensors(self.decoded_data(data))

    @override
    def system_inject_n_faults(self, data: Encoding, n: int):
        match data:
            case FullEncoding():
                tensor_list_fault_injection([data.encoded_bytes], n)
            case BitPatternEncoding():
                data.flip_n_bits(n)

    @override
    def system_metadata(self) -> dict[str, str]:
        metadata = copy.deepcopy(self.inner.system_metadata())

        match self.format:
            case EncodingFormatFull(chunk_size):
                metadata["bit_pattern"] = "all"
                metadata["chunk_size"] = str(chunk_size)
            case EncodingFormatBitPattern(pattern, pattern_length, bits_per_chunk):
                metadata["bit_pattern"] = f"{pattern}:{pattern_length}"
                metadata["chunk_size"] = str(bits_per_chunk)

        return metadata

    @override
    def system_clone_data(self, data: Encoding) -> Encoding:
        match data:
            case FullEncoding():
                return copy.deepcopy(data)
            case BitPatternEncoding():
                return data.clone()
