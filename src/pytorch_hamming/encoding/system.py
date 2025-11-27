import copy
import logging
from typing import override

from torch import Tensor

from pytorch_hamming.encoding.encoding import Encoder, Encoding
from pytorch_hamming.system import BaseSystem

logger = logging.getLogger(__name__)


class EncodedSystem[T](BaseSystem[Encoding]):
    def __init__(
        self,
        base: BaseSystem[T],
        encoder: Encoder,
    ) -> None:
        self.base: BaseSystem[T] = base
        self.encoded_data: Encoding | None = None
        self.encoder: Encoder = encoder

    def encode_base(self) -> Encoding:
        logger.debug("Encoding data tensors")
        data_tensors = self.base.system_data_tensors(self.base.system_data())
        return self.encoder.encode_tensor_list(data_tensors)

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

        self.encoder.add_metadata(metadata)

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
