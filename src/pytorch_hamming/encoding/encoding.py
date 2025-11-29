import abc
from typing import Self

from torch import Tensor


class Encoder(abc.ABC):
    @abc.abstractmethod
    def encode_tensor_list(self, ts: list[Tensor]) -> Encoding:
        """Encode a list of tensors."""
        ...

    def add_metadata(self, metadata: dict[str, str]) -> None:
        """Add metadata related to the encoding."""
        _ = metadata
        pass


class Encoding(abc.ABC):
    @abc.abstractmethod
    def decode_tensor_list(self) -> list[Tensor]:
        """Decode and return the list of tensors.

        Returns the tensors with the same shape as the original unencoded data.
        """
        ...

    @abc.abstractmethod
    def clone(self) -> Self:
        """Return a full clone of self.

        Modifying the clone should not modify the original in any way.
        """
        ...

    @abc.abstractmethod
    def flip_n_bits(self, n: int) -> None:
        """Flip a number of bits in the encoded data."""
        ...

    @abc.abstractmethod
    def bits_count(self) -> int:
        """Return the number of bits used for the encoded data."""
        ...
