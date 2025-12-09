import abc
from typing import Self

from torch import Tensor


class Encoder(abc.ABC):
    @abc.abstractmethod
    def encoder_encode_tensor_list(self, ts: list[Tensor]) -> Encoding:
        """Encode a list of tensors."""
        ...

    def encoder_add_metadata(self, metadata: dict[str, str]) -> None:
        """Add metadata related to the encoding."""
        _ = metadata
        pass


class Encoding(abc.ABC):
    @abc.abstractmethod
    def encoding_decode_tensor_list(self) -> list[Tensor]:
        """Decode and return the list of tensors.

        Returns the tensors with the same shape as the original unencoded data.
        """
        ...

    @abc.abstractmethod
    def encoding_clone(self) -> Self:
        """Return a full clone of self.

        Modifying the clone should not modify the original in any way.
        """
        ...

    @abc.abstractmethod
    def encoding_flip_n_bits(self, n: int) -> None:
        """Flip a number of bits in the encoded data."""
        ...

    @abc.abstractmethod
    def encoding_bits_count(self) -> int:
        """Return the number of bits used for the encoded data."""
        ...
