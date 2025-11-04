import copy
import logging
from dataclasses import dataclass

from pytorch_hamming.tensor_ops import tensor_list_fault_injection
import torch
from typing_extensions import override

from .._system import BaseSystem
from ._full import FullEncoding

logger = logging.getLogger(__name__)


@dataclass
class EncodingFormatFull:
    bits_per_chunk: int


# NOTE: the type alias is here because more types will be added later.
type EncodingFormat = EncodingFormatFull

# NOTE: the type alias is here because more types will be added later.
type Encoding = FullEncoding


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
                data = self.base.system_data_tensors(self.base.system_data())
                return FullEncoding.encode_tensor_list(data, bits_per_chunk)

    def decoded_base_data(self, data: Encoding) -> T:
        if self._decoded_cache is not None:
            logger.debug("Using cached value for decoded data")
            return self._decoded_cache

        logger.debug("Decoding data")
        base_data_copy = copy.deepcopy(self.base.system_data())
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
        # NOTE: we need to reset the decoded cache because the true values can
        # change if the values in the returned list are updated like through
        # fault injection.
        self._decoded_cache = None

        match data:
            case FullEncoding():
                tensor_list_fault_injection([data.encoded_bytes], n)

    @override
    def system_metadata(self) -> dict[str, str]:
        metadata = copy.deepcopy(self.base.system_metadata())

        match self.format:
            case EncodingFormatFull(chunk_size):
                metadata["bit_pattern"] = "all"
                metadata["chunk_size"] = str(chunk_size)

        return metadata
