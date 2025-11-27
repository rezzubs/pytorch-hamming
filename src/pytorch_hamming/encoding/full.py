from __future__ import annotations

import copy
import typing
from dataclasses import dataclass
from typing import Self, override

import hamming_core
import numpy as np
import torch

from pytorch_hamming.dtype import DnnDtype
from pytorch_hamming.encoding.encoding import Encoder, Encoding
from pytorch_hamming.tensor_ops import tensor_list_dtype, tensor_list_fault_injection


@dataclass
class FullEncoder(Encoder):
    bits_per_chunk: int

    @override
    def encode_tensor_list(self, ts: list[torch.Tensor]) -> Encoding:
        dtype = tensor_list_dtype(ts)
        if dtype is None:
            raise ValueError("Cannot encode an empty buffer")

        match DnnDtype.from_torch(dtype):
            case DnnDtype.Float32:
                with torch.no_grad():
                    rust_input = [t.flatten().numpy(force=True) for t in ts]
                    numpy_bytes, encoded_bits_count = hamming_core.encode_full_f32(
                        rust_input, self.bits_per_chunk
                    )
            case DnnDtype.Float16:
                with torch.no_grad():
                    rust_input = [
                        t.flatten().view(torch.uint16).numpy(force=True) for t in ts
                    ]
                    numpy_bytes, encoded_bits_count = hamming_core.encode_full_u16(
                        rust_input, self.bits_per_chunk
                    )

        # HACK: There's nothing we can do about this warning without an upstream fix.
        torch_bytes = torch.from_numpy(numpy_bytes)  # pyright: ignore[reportUnknownMemberType]
        assert torch_bytes.dtype == torch.uint8

        return FullEncoding(torch_bytes, self.bits_per_chunk, encoded_bits_count)

    @override
    def add_metadata(self, metadata: dict[str, str]) -> None:
        metadata["chunk_size"] = str(self.bits_per_chunk)


@dataclass
class FullEncoding(Encoding):
    _encoded_bytes: torch.Tensor
    _bits_per_chunk: int
    _bits_count: int

    @override
    def decode_tensor_list(self, output_buffer: list[torch.Tensor]) -> None:
        dtype = tensor_list_dtype(output_buffer)
        if dtype is None:
            raise ValueError("Cannot decode into an empty buffer")

        element_counts = [t.numel() for t in output_buffer]
        with torch.no_grad():
            numpy_bytes = self._encoded_bytes.numpy(force=True)
            assert numpy_bytes.dtype == np.dtype(np.uint8)
            numpy_bytes = typing.cast(np.typing.NDArray[np.uint8], numpy_bytes)

        match DnnDtype.from_torch(dtype):
            case DnnDtype.Float32:
                decoded, ded_results = hamming_core.decode_full_f32(
                    numpy_bytes,
                    self._bits_count,
                    self._bits_per_chunk,
                    element_counts,
                )
                # HACK: There's nothing we can do about this warning without an upstream fix.
                torch_decoded = [
                    torch.from_numpy(t)  # pyright: ignore[reportUnknownMemberType]
                    for t in decoded
                ]

            case DnnDtype.Float16:
                decoded, ded_results = hamming_core.decode_full_u16(
                    numpy_bytes,
                    self._bits_count,
                    self._bits_per_chunk,
                    element_counts,
                )
                torch_decoded = [
                    torch.from_numpy(t).view(torch.float16)  # pyright: ignore[reportUnknownMemberType]
                    for t in decoded
                ]

        for original, decoded in zip(output_buffer, torch_decoded, strict=True):
            # Discard because it's updated inplace.
            with torch.no_grad():
                _ = original.flatten().copy_(decoded)

        # TODO: we discard the double error detection results for now but may
        # want to do something with them in the future.
        _ = ded_results

    @override
    def clone(self) -> Self:
        return copy.deepcopy(self)

    @override
    def flip_n_bits(self, n: int):
        tensor_list_fault_injection([self._encoded_bytes], n)

    @override
    def bits_count(self) -> int:
        return self._bits_count
