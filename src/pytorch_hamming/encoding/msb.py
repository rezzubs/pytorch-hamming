"""Protection for a most significant bit"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Self, override

import hamming_core
import torch
from torch import Tensor

from pytorch_hamming.encoding.encoding import Encoder, Encoding
from pytorch_hamming.tensor_ops import tensor_list_dtype, tensor_list_fault_injection
from pytorch_hamming.utils import dtype_bits_count


class MsbEncoder(Encoder):
    @override
    def encode_tensor_list(self, ts: list[Tensor]) -> Encoding:
        dtype = tensor_list_dtype(ts)

        match dtype:
            case None:
                raise ValueError("Cannot encode an empty list")
            case torch.float32:
                pass
            case _:
                raise ValueError(
                    f"Cannot encode dtype={dtype} tensors. Only float32 is supported at this point"
                )

        bits_count = 0
        data: list[Tensor] = []
        item_bits_count = dtype_bits_count(dtype)

        for t in ts:
            bits_count += t.numel() * item_bits_count
            with torch.no_grad():
                t_np = t.numpy(force=True)
            hamming_core.bit30_encode_f32(t_np)
            t_encoded = torch.from_numpy(t_np)  # pyright: ignore[reportUnknownMemberType]
            data.append(t_encoded)

        return MsbEncoding(data, bits_count)

    @override
    def add_metadata(self, metadata: dict[str, str]) -> None:
        metadata["msb_duplicated"] = "true"


@dataclass
class MsbEncoding(Encoding):
    """An encoding format for protecting the most significant bits of parameter tensors.

    Most significant in this case referres to the second highest bit — the one
    after the sign bit. That bit will be copied to the lowest bits and those
    will be used to recover from single-bit errors.
    """

    _data: list[Tensor]
    _bits_count: int

    @override
    def decode_tensor_list(self, output_buffer: list[Tensor]) -> None:
        for original, encoded in zip(output_buffer, self._data, strict=True):
            with torch.no_grad():
                encoded_np = encoded.numpy(force=True)
                hamming_core.bit30_decode_f32(encoded_np)
                decoded = torch.from_numpy(encoded_np)  # pyright: ignore[reportUnknownMemberType]
                _ = original.copy_(decoded)

    @override
    def clone(self) -> Self:
        return deepcopy(self)

    @override
    def flip_n_bits(self, n: int) -> None:
        tensor_list_fault_injection(self._data, n)

    @override
    def bits_count(self) -> int:
        return self._bits_count
