from __future__ import annotations

import copy
import typing
from dataclasses import dataclass

import hamming_core
import numpy as np
import torch

from .._dtype import DnnDtype
from ..tensor_ops import tensor_list_dtype, tensor_list_fault_injection


@dataclass
class FullEncoding:
    _encoded_bytes: torch.Tensor
    _bits_per_chunk: int
    _bits_count: int

    @classmethod
    def encode_tensor_list(
        cls, ts: list[torch.Tensor], bits_per_chunk: int
    ) -> FullEncoding:
        dtype = tensor_list_dtype(ts)
        if dtype is None:
            raise ValueError("Cannot encode an empty buffer")

        match DnnDtype.from_torch(dtype):
            case DnnDtype.Float32:
                with torch.no_grad():
                    rust_input = [t.flatten().numpy(force=True) for t in ts]
                    numpy_bytes, encoded_bits_count = hamming_core.encode_full_f32(
                        rust_input, bits_per_chunk
                    )
            case DnnDtype.Float16:
                with torch.no_grad():
                    rust_input = [
                        t.flatten().view(torch.uint16).numpy(force=True) for t in ts
                    ]
                    numpy_bytes, encoded_bits_count = hamming_core.encode_full_u16(
                        rust_input, bits_per_chunk
                    )

        # HACK: There's nothing we can do about this warning without an upstream fix.
        torch_bytes = torch.from_numpy(numpy_bytes)  # pyright: ignore[reportUnknownMemberType]
        assert torch_bytes.dtype == torch.uint8

        return cls(torch_bytes, bits_per_chunk, encoded_bits_count)

    def decode_tensor_list(
        self, output_buffer: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Decode into the output buffer.

        `output_buffer` should have the same structure as the original data.

        Returns a reference to the same output buffer.
        """
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

        return output_buffer

    def clone(self) -> FullEncoding:
        return copy.deepcopy(self)

    def flip_n_bits(self, n: int):
        tensor_list_fault_injection([self._encoded_bytes], n)

    def bits_count(self) -> int:
        return self._bits_count
