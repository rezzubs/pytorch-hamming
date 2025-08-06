"""Wrappers for the rust hamming module"""

import numpy
import torch
from torch import nn

import hamming

__all__ = ["hamming_encode64", "hamming_decode64"]


def hamming_encode64(t: torch.Tensor) -> torch.Tensor:
    """Enocde a flattened tensor as 9 byte hamming codes.

    Returns:
        A 1 dimensional tensor with dtype=uint8

    Note that encoding adds an extra 0 for odd length tensors which needs to be
    removed manually after decoding.
    """
    if len(t.shape) != 1:
        raise ValueError(f"Expected a flattened tensor, got shape {t.shape}")

    # TODO: match on dtype and add support for f16-f64.
    if t.dtype != torch.float32:
        raise ValueError(f"Only float32 tensors are supported, got {t.dtype}")

    # FIXME: Ignored because there are no type signatures for the hamming module.
    out: numpy.ndarray = hamming.encode64(t.numpy())  # pyright: ignore

    return torch.from_numpy(out)


def hamming_decode64(t: torch.Tensor) -> torch.Tensor:
    """Decode the output of `hamming_encode64`.

    Returns:
        A 1 dimensional tensor with dtype=float32

    Note that encoding adds an extra 0 for odd length tensors which needs to be
    removed manually after decoding.
    """
    if len(t.shape) != 1:
        raise ValueError(f"Expected a flattened tensor, got shape {t.shape}")

    if t.dtype != torch.uint8:
        raise ValueError(f"Expected dtype=uint8, got {t.dtype}")

    # NOTE: Length checks are handled in rust.
    # FIXME: Ignored because there are no type signatures for the hamming module.
    out: numpy.ndarray = hamming.decode64(t.numpy())  # pyright: ignore

    return torch.from_numpy(out)


type SupportsHamming = nn.Linear | nn.Conv2d | nn.BatchNorm2d


class HammingLayer(nn.Module):
    """A wrapper for layers in a neural network which encodes the weights as hamming codes.

    Must be decoded before usage.
    """

    def __init__(self, original: SupportsHamming, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.original = original
        self.register_buffer("original_shape", torch.tensor(original.weight.shape))
        self.register_buffer("original_len", torch.tensor(original.weight.numel()))
        protected_data = hamming_encode64(self.original.weight.data.flatten())
        self.register_buffer("protected_data", protected_data)

    def decode(self) -> SupportsHamming:
        shape_tensor = self.get_buffer("original_shape")
        shape = torch.Size(shape_tensor.tolist())
        length = self.get_buffer("original_len").item()

        protected_data = self.get_buffer("protected_data")
        weight = hamming_decode64(protected_data)[:length]
        self.original.weight.data = weight.reshape(shape)
        return self.original

    def forward(self) -> None:
        raise RuntimeError("Hamming layers need to be decoded before usage.")
