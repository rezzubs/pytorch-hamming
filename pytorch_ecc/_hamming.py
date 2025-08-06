"""Wrappers for the rust hamming module"""

import numpy
import torch

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
        raise ValueError(f"Expected a flattened tensor, got {t.shape}")

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
        raise ValueError(f"Expected a flattened tensor, got {t.shape}")

    if t.dtype != torch.uint8:
        raise ValueError(f"Expected dtype=uint8, got {t.dtype}")

    # NOTE: Length checks are handled in rust.
    # FIXME: Ignored because there are no type signatures for the hamming module.
    out: numpy.ndarray = hamming.decode64(t.numpy())  # pyright: ignore

    return torch.from_numpy(out)
