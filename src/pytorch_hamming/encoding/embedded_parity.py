import logging
from dataclasses import dataclass
from typing import override

import hamming_core
import torch
from torch import Tensor

from pytorch_hamming.encoding.sequence import TensorEncoder, TensorEncoding
from pytorch_hamming.tensor_ops import tensor_list_dtype, tensor_list_fault_injection
from pytorch_hamming.utils import dtype_bits_count

_logger = logging.getLogger(__name__)


class EmbeddedParityEncoder(TensorEncoder):
    @override
    def tensor_encoder_encode_tensor_list(
        self, ts: list[Tensor]
    ) -> EmbeddedParityEncoding:
        dtype = tensor_list_dtype(ts)

        match dtype:
            case None:
                raise ValueError("Cannot encode an empty list")
            case torch.float32:

                def encode(t: Tensor) -> Tensor:
                    with torch.no_grad():
                        t_np = t.numpy(force=True)
                    hamming_core.embedded_parity_encode_f32(t_np)
                    return torch.from_numpy(t_np)  # pyright: ignore[reportUnknownMemberType]
            case torch.float16:

                def encode(t: Tensor) -> Tensor:
                    with torch.no_grad():
                        t_np = t.view(torch.uint16).numpy(force=True)
                    hamming_core.embedded_parity_encode_u16(t_np)
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

        return EmbeddedParityEncoding(data, bits_count, decoded_tensors, dtype)

    @override
    def encoder_add_metadata(self, metadata: dict[str, str]) -> None:
        metadata["embedded_parity"] = "true"


@dataclass
class EmbeddedParityEncoding(TensorEncoding):
    _encoded_data: list[Tensor]
    _bits_count: int
    _decoded_tensors: list[Tensor]
    _dtype: torch.dtype
    _needs_recompute: bool = True

    @override
    def tensor_encoding_tensors(self) -> list[Tensor]:
        return self._encoded_data

    @override
    def tensor_encoding_trigger_recompute(self) -> None:
        self._needs_recompute = True

    @override
    def encoding_decode_tensor_list(self) -> list[Tensor]:
        if not self._needs_recompute:
            return self._decoded_tensors

        match self._dtype:
            case torch.float16:

                def decode(t: Tensor) -> Tensor:
                    encoded_np = t.view(torch.uint16).numpy(force=True)
                    hamming_core.embedded_parity_decode_u16(encoded_np)
                    return torch.from_numpy(encoded_np).view(torch.float16)  # pyright: ignore[reportUnknownMemberType]

            case torch.float32:

                def decode(t: Tensor) -> Tensor:
                    encoded_np = t.numpy(force=True)
                    hamming_core.embedded_parity_decode_f32(encoded_np)
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
    def encoding_clone(self) -> EmbeddedParityEncoding:
        copied_data = [t.clone() for t in self._encoded_data]
        copied_decoded = [t.clone() for t in self._decoded_tensors]
        return EmbeddedParityEncoding(
            copied_data,
            self._bits_count,
            copied_decoded,
            self._dtype,
            self._needs_recompute,
        )

    @override
    def encoding_flip_n_bits(self, n: int) -> None:
        _logger.debug("Invalidating decoded tensors due to fault injection")
        self.tensor_encoding_trigger_recompute()
        tensor_list_fault_injection(self._encoded_data, n)

    @override
    def encoding_bits_count(self) -> int:
        return self._bits_count
