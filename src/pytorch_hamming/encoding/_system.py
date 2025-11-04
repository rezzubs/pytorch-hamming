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
        # self.EncodingStrategy = encoding_strategy
        self.base: BaseSystem[T] = base
        self.format: EncodingFormat = format
        self._encoded_cache: Encoding | None = None
        self._decoded_cache: T | None = copy.deepcopy(self.base.system_data())

    def encode_base(self) -> Encoding:
        logger.debug("Encoding data tensors")
        match self.format:
            case EncodingFormatFull(bits_per_chunk):
                data_tensors = self.base.system_data_tensors(self.base.system_data())
                return FullEncoding.encode_tensor_list(data_tensors, bits_per_chunk)
            case EncodingFormatBitPattern(pattern, pattern_length, bits_per_chunk):
                data_tensors = self.base.system_data_tensors(self.base.system_data())
                return BitPatternEncoding.encode_tensor_list(
                    ts=data_tensors,
                    pattern=pattern,
                    pattern_length=pattern_length,
                    bits_per_chunk=bits_per_chunk,
                )

    def decoded_base_data(self, data: Encoding) -> T:
        if self._decoded_cache is not None:
            logger.debug("Using cached value for decoded data")
            return self._decoded_cache

        logger.debug("Decoding data")
        base_data_copy = self.base.system_clone_data(self.base.system_data())
        base_data_copy_tensors = self.base.system_data_tensors(base_data_copy)

        # discard because it's updated in place.
        _ = data.decode_tensor_list(base_data_copy_tensors)

        self._decoded_cache = base_data_copy
        return base_data_copy

    @override
    def system_data(self) -> Encoding:
        if self._encoded_cache is None:
            self._encoded_cache = self.encode_base()

        return self._encoded_cache

    @override
    def system_accuracy(self, data: Encoding) -> float:
        return self.base.system_accuracy(self.decoded_base_data(data))

    @override
    def system_data_tensors(self, data: Encoding) -> list[torch.Tensor]:
        return self.base.system_data_tensors(self.decoded_base_data(data))

    @override
    def system_inject_n_faults(self, data: Encoding, n: int):
        # NOTE: we need to reset the decoded cache because the true values will
        # be altered for a nonzero value of `n`.
        self._decoded_cache = None

        match data:
            case FullEncoding():
                tensor_list_fault_injection([data.encoded_bytes], n)
            case BitPatternEncoding():
                data.flip_n_bits(n)

    @override
    def system_metadata(self) -> dict[str, str]:
        metadata = copy.deepcopy(self.base.system_metadata())

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
