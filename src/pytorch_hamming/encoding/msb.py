"""Protection for a most significant bit"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import override

import hamming_core
import torch
from torch import Tensor

from pytorch_hamming.encoding.encoding import Encoder, Encoding
from pytorch_hamming.tensor_ops import tensor_list_dtype, tensor_list_fault_injection
from pytorch_hamming.utils import dtype_bits_count

_logger = logging.getLogger(__name__)


def _add_metadata(metadata: dict[str, str]) -> None:
    metadata["msb_duplicated"] = "true"


class MsbEncoder(Encoder):
    @override
    def encode_tensor_list(self, ts: list[Tensor]) -> Encoding:
        dtype = tensor_list_dtype(ts)

        match dtype:
            case None:
                raise ValueError("Cannot encode an empty list")
            case torch.float32:

                def encode(t: Tensor) -> Tensor:
                    with torch.no_grad():
                        t_np = t.numpy(force=True)
                    hamming_core.bit30_encode_f32(t_np)
                    return torch.from_numpy(t_np)  # pyright: ignore[reportUnknownMemberType]
            case torch.float16:

                def encode(t: Tensor) -> Tensor:
                    with torch.no_grad():
                        t_np = t.view(torch.uint16).numpy(force=True)
                    hamming_core.bit14_encode_u16(t_np)
                    return torch.from_numpy(t_np).view(torch.float16)  # pyright: ignore[reportUnknownMemberType]
            case _:
                raise ValueError(f"Cannot encode dtype={dtype} tensors.")

        # Store decoded tensor copies
        decoded_tensors = [t.clone() for t in ts]

        bits_count = 0
        data: list[Tensor] = []
        item_bits_count = dtype_bits_count(dtype)

        for t in ts:
            bits_count += t.numel() * item_bits_count
            data.append(encode(t))

        return MsbEncoding(data, bits_count, decoded_tensors, dtype)

    @override
    def add_metadata(self, metadata: dict[str, str]) -> None:
        _add_metadata(metadata)


@dataclass
class MsbEncoding(Encoding):
    """An encoding format for protecting the most significant bits of parameter tensors.

    Most significant in this case referres to the second highest bit — the one
    after the sign bit. That bit will be copied to the lowest bits and those
    will be used to recover from single-bit errors.
    """

    _encoded_data: list[Tensor]
    _bits_count: int
    _decoded_tensors: list[Tensor]
    _dtype: torch.dtype
    _needs_recompute: bool = False

    @override
    def decode_tensor_list(self) -> list[Tensor]:
        if not self._needs_recompute:
            return self._decoded_tensors

        match self._dtype:
            case torch.float16:

                def decode(t: Tensor) -> Tensor:
                    encoded_np = t.view(torch.uint16).numpy(force=True)
                    hamming_core.bit14_decode_u16(encoded_np)
                    return torch.from_numpy(encoded_np).view(torch.float16)  # pyright: ignore[reportUnknownMemberType]

            case torch.float32:

                def decode(t: Tensor) -> Tensor:
                    encoded_np = t.numpy(force=True)
                    hamming_core.bit30_decode_f32(encoded_np)
                    return torch.from_numpy(encoded_np)  # pyright: ignore[reportUnknownMemberType]
            case _:
                raise RuntimeError(f"Unsupported dtype: {self._dtype}")

        for output, encoded in zip(
            self._decoded_tensors, self._encoded_data, strict=True
        ):
            with torch.no_grad():
                decoded = decode(encoded)
                _ = output.copy_(decoded)

        self._needs_recompute = False
        return self._decoded_tensors

    @override
    def clone(self) -> MsbEncoding:
        copied_data = [t.clone() for t in self._encoded_data]
        copied_decoded = [t.clone() for t in self._decoded_tensors]
        return MsbEncoding(
            copied_data,
            self._bits_count,
            copied_decoded,
            self._dtype,
            self._needs_recompute,
        )

    @override
    def flip_n_bits(self, n: int) -> None:
        _logger.debug("Invalidating decoded tensors due to fault injection")
        self._needs_recompute = True
        tensor_list_fault_injection(self._encoded_data, n)

    @override
    def bits_count(self) -> int:
        return self._bits_count


@dataclass
class MsbMixedEncoder(Encoder):
    """Apply another encoding on top of `MsbEncoding`"""

    encoder: Encoder

    @override
    def encode_tensor_list(self, ts: list[Tensor]) -> Encoding:
        msb_encoded = MsbEncoder().encode_tensor_list(ts)
        assert isinstance(msb_encoded, MsbEncoding)
        encoded_data = self.encoder.encode_tensor_list(
            msb_encoded._encoded_data  # pyright: ignore[reportPrivateUsage]
        )
        return MsbMixedEncoding(encoded_data, msb_encoded)

    @override
    def add_metadata(self, metadata: dict[str, str]) -> None:
        _add_metadata(metadata)
        self.encoder.add_metadata(metadata)


@dataclass
class MsbMixedEncoding(Encoding):
    _overlay: Encoding
    _base_data: MsbEncoding

    @override
    def decode_tensor_list(self) -> list[Tensor]:
        msb_data = self._overlay.decode_tensor_list()
        self._base_data._encoded_data = msb_data  # pyright: ignore[reportPrivateUsage]
        return self._base_data.decode_tensor_list()

    @override
    def clone(self) -> MsbMixedEncoding:
        return MsbMixedEncoding(self._overlay.clone(), self._base_data.clone())

    @override
    def flip_n_bits(self, n: int) -> None:
        self._base_data._needs_recompute = True  # pyright: ignore[reportPrivateUsage]
        self._overlay.flip_n_bits(n)

    @override
    def bits_count(self) -> int:
        return self._overlay.bits_count()
