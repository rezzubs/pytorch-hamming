from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import override

import hamming_core
import torch

from pytorch_hamming.dtype import DnnDtype
from pytorch_hamming.encoding.encoding import Encoder, Encoding
from pytorch_hamming.tensor_ops import tensor_list_dtype

_logger = logging.getLogger(__name__)


@dataclass
class FullEncoder(Encoder):
    bits_per_chunk: int

    @override
    def encoder_encode_tensor_list(self, ts: list[torch.Tensor]) -> Encoding:
        dtype = tensor_list_dtype(ts)
        if dtype is None:
            raise ValueError("Cannot encode an empty buffer")

        # Store decoded tensor copies
        decoded_tensors = [t.clone() for t in ts]

        match DnnDtype.from_torch(dtype):
            case DnnDtype.Float32:
                with torch.no_grad():
                    rust_input = [t.flatten().numpy(force=True) for t in ts]
                    encoded_data = hamming_core.encode_full_f32(
                        rust_input, self.bits_per_chunk
                    )
            case DnnDtype.Float16:
                with torch.no_grad():
                    rust_input = [
                        t.flatten().view(torch.uint16).numpy(force=True) for t in ts
                    ]
                    encoded_data = hamming_core.encode_full_u16(
                        rust_input, self.bits_per_chunk
                    )

        return FullEncoding(
            encoded_data,
            decoded_tensors,
            dtype,
        )

    @override
    def encoder_add_metadata(self, metadata: dict[str, str]) -> None:
        metadata["chunk_size"] = str(self.bits_per_chunk)


@dataclass
class FullEncoding(Encoding):
    _encoded_data: hamming_core.FullEncoding
    _decoded_tensors: list[torch.Tensor]
    _dtype: torch.dtype
    _needs_recompute: bool = False

    @override
    def encoding_decode_tensor_list(self) -> list[torch.Tensor]:
        if not self._needs_recompute:
            _logger.debug("Using cached decoded tensors")
            return self._decoded_tensors
        _logger.debug("Recomputing decoded tensors")

        match DnnDtype.from_torch(self._dtype):
            case DnnDtype.Float32:
                decoded, ded_results = self._encoded_data.decode_full_f32()
                # HACK: There's nothing we can do about this warning without an upstream fix.
                torch_decoded = [
                    torch.from_numpy(t)  # pyright: ignore[reportUnknownMemberType]
                    for t in decoded
                ]

            case DnnDtype.Float16:
                decoded, ded_results = self._encoded_data.decode_full_u16()
                torch_decoded = [
                    torch.from_numpy(t).view(torch.float16)  # pyright: ignore[reportUnknownMemberType]
                    for t in decoded
                ]

        # Update cached decoded tensors in-place
        for cached, decoded in zip(self._decoded_tensors, torch_decoded, strict=True):
            with torch.no_grad():
                _ = cached.flatten().copy_(decoded)

        # TODO: we discard the double error detection results for now but may
        # want to do something with them in the future.
        _ = ded_results

        self._needs_recompute = False
        return self._decoded_tensors

    @override
    def encoding_clone(self) -> FullEncoding:
        return FullEncoding(
            self._encoded_data.clone(),
            self._decoded_tensors,
            self._dtype,
            self._needs_recompute,
        )

    @override
    def encoding_flip_n_bits(self, n: int):
        _logger.debug("Invalidating decoded tensor cache due to fault injection.")
        self._needs_recompute = True
        self._encoded_data.flip_n_bits(n)

    @override
    def encoding_bits_count(self) -> int:
        return self._encoded_data.bits_count()
