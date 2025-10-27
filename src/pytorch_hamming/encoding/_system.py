from dataclasses import dataclass
from typing import Generic, TypeVar, TypeAlias
import copy
from typing_extensions import override
from .._system import BaseSystem
from ._full import EncodingFull
import torch


@dataclass
class EncodingFormatFull:
    bits_per_chunk: int


# NOTE: the type alias is here because more types will be added later.
EncodingFormat: TypeAlias = EncodingFormatFull

# NOTE: the type alias is here because more types will be added later.
Encoding: TypeAlias = EncodingFull

T = TypeVar("T")


class EncodedSystem(BaseSystem[Encoding], Generic[T]):
    def __init__(
        self,
        base: BaseSystem[T],
        format: EncodingFormat,
    ) -> None:
        # self.EncodingStrategy = encoding_strategy
        self.base: BaseSystem[T] = base
        self.format: EncodingFormat = format
        self._encoded_cache: Encoding | None = None

    def encode_base(self) -> Encoding:
        match self.format:
            case EncodingFormatFull(bits_per_chunk):
                data = self.base.system_data_tensors(self.base.system_data())
                return EncodingFull.encode_tensor_list(data, bits_per_chunk)

    @override
    def system_data(self) -> Encoding:
        if self._encoded_cache is None:
            self._encoded_cache = self.encode_base()

        return self._encoded_cache

    @override
    def system_accuracy(self, data: Encoding) -> float:
        base_data_copy = copy.deepcopy(self.base.system_data())
        base_data_copy_tensors = self.base.system_data_tensors(self.base.system_data())

        # discard because it's updated in place.
        _ = data.decode_tensor_list(base_data_copy_tensors)

        return self.base.system_accuracy(base_data_copy)

    @override
    def system_data_tensors(self, data: Encoding) -> list[torch.Tensor]:
        match data:
            case EncodingFull():
                return [data.encoded_bytes]

    @override
    def system_metadata(self) -> dict[str, str]:
        metadata = copy.deepcopy(self.base.system_metadata())

        match self.format:
            case EncodingFormatFull(chunk_size):
                metadata["bit_pattern"] = "all"
                metadata["chunk_size"] = str(chunk_size)

        return metadata
