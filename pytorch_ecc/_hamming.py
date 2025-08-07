"""Utilities for encoding PyTorch modules with hamming codes."""

import random

import numpy
import torch
from torch import nn

import hamming

__all__ = [
    "HammingLayer",
    "hamming_decode64",
    "hamming_decode_module",
    "hamming_encode64",
    "hamming_encode_module",
    "hamming_layer_fi",
]

HAMMING_DATA_PREFIX = "hamming_protected_"
BITS_PER_BYTE = 8


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


SupportsHamming = nn.Linear | nn.Conv2d | nn.BatchNorm2d


class HammingLayer(nn.Module):
    """A wrapper for layers in a neural network which encodes the weights as hamming codes.

    Must be decoded before usage.
    """

    def __init__(self, original: SupportsHamming, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(original, SupportsHamming):
            raise ValueError(
                f"Module {type(original)} is not a valid HammingLayer target"
            )

        self.original = original

        self._protect_tensor("weight", original.weight.data)

        if original.bias is not None:
            self._protect_tensor("bias", original.bias.data)

        if not isinstance(original, nn.BatchNorm2d):
            return

        if original.running_mean is not None:
            if isinstance(original.running_mean, nn.Module):
                raise ValueError("Unsupported module type")

            self._protect_tensor("running_mean", original.running_mean.data)

        if original.running_var is not None:
            if isinstance(original.running_var, nn.Module):
                raise ValueError("Unsupported module type")

            self._protect_tensor("running_var", original.running_var.data)

    def _protect_tensor(self, name: str, t: torch.Tensor) -> None:
        """Protect a parameter by encoding it as a hamming code.

        These parameters will be used for fault injection.

        See also `_decode_protected`.
        """
        og = "hamming_original_" + name
        t = t.data

        protected_data = hamming_encode64(t.flatten())
        self.register_buffer(HAMMING_DATA_PREFIX + name, protected_data)

        self.register_buffer(og + "_shape", torch.tensor(t.shape))

        self.register_buffer(og + "_len", torch.tensor(t.numel()))

    def _decode_protected(self, name: str) -> torch.Tensor:
        """Decode a protected named parameter.

        See also `_protect_tensor`
        """
        og = "hamming_original_" + name

        protected_data = self.get_buffer(HAMMING_DATA_PREFIX + name)

        shape_tensor = self.get_buffer(og + "_shape")
        shape = torch.Size(shape_tensor.tolist())

        length = self.get_buffer(og + "_len").item()

        return hamming_decode64(protected_data)[:length].reshape(shape)

    def decode(self) -> SupportsHamming:
        """Decode the hamming module into the type it was initialized with.

        Using the hamming module after decoding is undefined behavior.
        """
        self.original.weight.data = self._decode_protected("weight")

        if self.original.bias is not None:
            self.original.bias.data = self._decode_protected("bias")

        if not isinstance(self.original, nn.BatchNorm2d):
            return self.original

        if self.original.running_mean is not None:
            self.original.running_mean = self._decode_protected("running_mean")

        if self.original.running_var is not None:
            self.original.running_var = self._decode_protected("running_var")

        return self.original

    def forward(self) -> None:
        raise RuntimeError(
            "Hamming layers need to be decoded before usage. See `hamming_decode_module`"
        )


def hamming_encode_module(module: nn.Module) -> None:
    """Recursively replace child layers of the module with `HammingLayer`

    A module that has been prepared like this can be used as an input for
    `hamming_layer_fi` for fault injection.

    Use `hamming_decode_module` to restore the original representation.

    See `SupportsHamming` for supported layer types.
    """
    for name, child in module.named_children():
        hamming_encode_module(child)

        if not isinstance(child, SupportsHamming):
            continue

        setattr(module, name, HammingLayer(child))


def hamming_decode_module(module: nn.Module) -> None:
    """Decodes all `HammingLayer` children into their original instances.

    This corrects all single bit errors in a memory line caused by `hamming_layer_fi`.

    See `hamming_encode_module`.
    """
    for name, child in module.named_children():
        hamming_decode_module(child)

        if not isinstance(child, HammingLayer):
            continue

        setattr(module, name, child.decode())


def tensor_flip_bit(t: torch.Tensor, bit_index: int) -> None:
    """Flip a single bit in a uint8 tensor.

    The values in the tensor are treated as a continuous stream of bits.
    """
    if t.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 tensor, got {t.dtype}")
    if len(t.shape) != 1:
        raise ValueError(f"Expected a single dimensional tensor, got shape {t.shape}")

    num_bits = t.numel() * BITS_PER_BYTE

    if bit_index >= num_bits:
        raise ValueError(f"Tensor has {num_bits} bits, got index {bit_index}")

    byte_index = bit_index // BITS_PER_BYTE
    true_bit_index = bit_index % BITS_PER_BYTE

    t[byte_index] = t[byte_index] ^ (1 << true_bit_index)


def tensor_list_flip_bit(ts: list[torch.Tensor], bit_index: int) -> None:
    """Flip a single bit in a list of tensors.

    The list of tensors are interpreted as a continuous stream of bits.
    """
    start_bit = 0
    for t in ts:
        num_bits = t.numel() * BITS_PER_BYTE
        first_bit_of_next = start_bit + num_bits

        if first_bit_of_next <= bit_index:
            start_bit = first_bit_of_next
            continue

        t_bit_index = bit_index - start_bit
        assert t_bit_index >= 0

        tensor_flip_bit(t, t_bit_index)
        return

    total_num_bits = sum([t.numel() * BITS_PER_BYTE for t in ts])
    raise ValueError(
        f"Tensor list has {total_num_bits} bits in total, got index {bit_index}"
    )


def hamming_layer_fi(module: nn.Module, num_flips: int) -> None:
    """Inject faults uniformly into `HammingLayer` children of the module.

    All bit flips will be unique.

    See `hamming_encode_module`.
    """
    protected_buffers = list(
        x[1]
        for x in module.named_buffers(recurse=True, remove_duplicate=False)
        if HAMMING_DATA_PREFIX in x[0]
    )

    total_num_bits = sum([t.numel() * BITS_PER_BYTE for t in protected_buffers])
    print(total_num_bits)
    if total_num_bits < num_flips:
        raise ValueError(
            f"The module has {total_num_bits} bits worth of unprotected data, "
            "tried to inject {num_flips} faults"
        )

    flip_candidates = list(range(total_num_bits))
    random.shuffle(flip_candidates)

    for _ in range(num_flips):
        bit_to_flip = flip_candidates.pop()

        tensor_list_flip_bit(protected_buffers, bit_to_flip)
