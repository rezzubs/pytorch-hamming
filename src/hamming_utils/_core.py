"""Utilities for encoding PyTorch modules with hamming codes."""

from __future__ import annotations

import numpy
import torch
from torch import nn

import hamming_core

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._stats import HammingStats

DATA_PREFIX = "hamming_protected_"
ORIGINAL_PREFIX = "hamming_original_"
DTYPE_F32 = 0
DTYPE_F16 = 1

BITS_PER_BYTE = 8
BYTES_PER_CONTAINER = 9
BITS_PER_CONTAINER = BITS_PER_BYTE * BYTES_PER_CONTAINER


def encode_f32(t: torch.Tensor) -> torch.Tensor:
    """Enocde a flattened float32 tensor as 9 byte hamming codes.

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
    out: numpy.ndarray = hamming_core.u64.encode_f32(t.numpy())  # pyright: ignore

    return torch.from_numpy(out)


def decode_f32(t: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Decode the output of `encode_f32`.

    Returns:
        A 1 dimensional tensor with dtype=float32 and the number of faults that
        could not be corrected but were still detected.

    Note that encoding adds an extra 0 for odd length tensors which needs to be
    removed manually after decoding.
    """
    if len(t.shape) != 1:
        raise ValueError(f"Expected a flattened tensor, got shape {t.shape}")

    if t.dtype != torch.uint8:
        raise ValueError(f"Expected dtype=uint8, got {t.dtype}")

    # NOTE: Length checks are handled in rust.
    # FIXME: Ignored because there are no type signatures for the hamming module.
    result: tuple[numpy.ndarray, int] = hamming_core.u64.decode_f32(t.numpy())  # pyright: ignore

    return torch.from_numpy(result[0]), result[1]


def encode_f16(t: torch.Tensor) -> torch.Tensor:
    """Enocde a flattened float32 tensor as 9 byte hamming codes.

    Returns:
        A 1 dimensional tensor with dtype=uint8

    Note that encoding adds padding zeros to make the length a multiple of 4.
    These need to be removed manually after decoding.
    """
    if len(t.shape) != 1:
        raise ValueError(f"Expected a flattened tensor, got shape {t.shape}")

    if t.dtype != torch.float16:
        raise ValueError(f"Expected dtype=float16, got {t.dtype}")

    result: numpy.ndarray = hamming_core.u64.encode_u16(t.view(torch.uint16).numpy())  # pyright: ignore
    torch_result = torch.from_numpy(result)
    assert torch_result.dtype == torch.uint8

    return torch_result


def decode_f16(t: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Decode the output of `encode_f16`.

    Returns:
        A 1 dimensional tensor with dtype=float16 and the number of faults that
        could not be corrected but were still detected.

    Note that encoding adds padding zeros to make the length a multiple of 4.
    These need to be removed manually after decoding.
    """
    if len(t.shape) != 1:
        raise ValueError(f"Expected a flattened tensor, got shape {t.shape}")

    if t.dtype != torch.uint8:
        raise ValueError(f"Expected dtype=uint8, got {t.dtype}")

    # NOTE: Length checks are handled in rust.
    # FIXME: Ignored because there are no type signatures for the hamming module.
    result: tuple[numpy.ndarray, int] = hamming_core.u64.decode_u16(t.numpy())  # pyright: ignore
    torch_result = torch.from_numpy(result[0])
    assert torch_result.dtype == torch.uint16

    return torch_result.view(torch.float16), result[1]


def tensor_list_fi(
    ts: list[torch.Tensor], bit_error_rate: float, *, stats: HammingStats | None = None
) -> None:
    """Inject faults uniformly in a list of tensors by the given bit error rate."""
    if not (0 <= bit_error_rate <= 1):
        raise ValueError("Bit error rate must be between 0 and 1 inclusive")

    for i, t in enumerate(ts):
        if t.dtype != torch.uint8:
            raise ValueError(f"Expected dtype=uint8, got {t.dtype} (tensor {i}")

    flattened = [t.flatten() for t in ts]

    # FIXME: Ignored because there are no type signatures for the hamming module.
    (result, context) = hamming_core.u64.array_list_fi(  # type: ignore
        [t.numpy() for t in flattened], bit_error_rate
    )
    assert isinstance(result, list)
    assert isinstance(context, dict)

    if stats is not None:
        stats.num_faults = context["num_faults"]
        stats.total_bits = context["total_bits"]

    for old, new in zip(flattened, result, strict=True):
        new = torch.from_numpy(new)
        assert new.dtype == torch.uint8
        old.copy_(new)


SupportsHamming = nn.Linear | nn.Conv2d | nn.BatchNorm2d


def compare_parameter_bitwise(a: torch.Tensor, b: torch.Tensor) -> list[int]:
    assert a.shape == b.shape
    assert a.dtype == b.dtype

    if a.dtype == torch.float32:
        view_repr = torch.uint32
    elif a.dtype == torch.float16:
        view_repr = torch.uint16
    else:
        raise ValueError(f"Unsupported dtype {a.dtype}")

    out = []
    for a_item, b_item in zip(a.flatten(), b.flatten(), strict=True):
        a_bits = int(a_item.view(view_repr).item())
        b_bits = int(b_item.view(view_repr).item())

        xor = a_bits ^ b_bits
        # NOTE: != because the most significant bit of i32 is the sign bit,
        # therefore we need to account for negative values.
        if xor != 0:
            out.append(xor)
    return out


def compare_module_bitwise(a: nn.Module, b: nn.Module) -> list[int]:
    """Recursively compare `SupportsHamming` children and return a bitwise xor of non-matching items.

    The modules are expected to have an identical representation.
    """
    out = []
    for a_child, b_child in zip(a.children(), b.children(), strict=True):
        out += compare_module_bitwise(a_child, b_child)

    if not isinstance(a, SupportsHamming):
        assert not isinstance(b, SupportsHamming)
        return out
    if not isinstance(b, SupportsHamming):
        assert not isinstance(a, SupportsHamming)
        return out

    out += compare_parameter_bitwise(a.weight, b.weight)
    if a.bias is not None:
        assert b.bias is not None
        out += compare_parameter_bitwise(a.bias, b.bias)

    if not isinstance(a, nn.BatchNorm2d):
        assert not isinstance(b, nn.BatchNorm2d)
        return out

    if a.running_mean is not None:
        assert b.running_mean is not None
        assert not isinstance(a.running_mean, nn.Module)
        assert not isinstance(b.running_mean, nn.Module)
        out += compare_parameter_bitwise(a.running_mean, b.running_mean)

    if a.running_var is not None:
        assert b.running_var is not None
        assert not isinstance(a.running_var, nn.Module)
        assert not isinstance(b.running_var, nn.Module)
        out += compare_parameter_bitwise(a.running_var, b.running_var)

    return out


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

        self._original = original
        # Used as a shared buffer during decoding
        self._failed_decodings = 0

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
        og = ORIGINAL_PREFIX + name
        t = t.data

        if t.dtype == torch.float32:
            dtype = DTYPE_F32
            protected_data = encode_f32(t.flatten())
        elif t.dtype == torch.float16:
            dtype = DTYPE_F16
            protected_data = encode_f16(t.flatten())
        else:
            raise ValueError(f"Unsupported datatype {t.dtype}")
        self.register_buffer(DATA_PREFIX + name, protected_data)

        self.register_buffer(og + "_shape", torch.tensor(t.shape))
        self.register_buffer(og + "_len", torch.tensor(t.numel()))
        self.register_buffer(og + "_dtype", torch.tensor(dtype))

    def _decode_protected(self, name: str) -> torch.Tensor:
        """Decode a protected named parameter.

        See also `_protect_tensor`
        """
        og = ORIGINAL_PREFIX + name

        protected_data = self.get_buffer(DATA_PREFIX + name)

        shape_tensor = self.get_buffer(og + "_shape")
        shape = torch.Size(shape_tensor.tolist())

        length = self.get_buffer(og + "_len").item()
        dtype = self.get_buffer(og + "_dtype").item()

        if dtype == DTYPE_F32:
            result = decode_f32(protected_data)
        elif dtype == DTYPE_F16:
            result = decode_f16(protected_data)
        else:
            raise ValueError(f"Unexpected dtype variant `{dtype}`")

        self._failed_decodings += result[1]

        return result[0][:length].reshape(shape)

    def decode(self) -> tuple[SupportsHamming, int]:
        """Decode the hamming module into the type it was initialized with.

        Using the hamming module after decoding is undefined behavior.

        Returns:
            The original layer and the number of containers which couldn't be
            corrected.
        """
        self._original.weight.data = self._decode_protected("weight")

        if self._original.bias is not None:
            self._original.bias.data = self._decode_protected("bias")

        if not isinstance(self._original, nn.BatchNorm2d):
            return self._original, self._failed_decodings

        if self._original.running_mean is not None:
            self._original.running_mean = self._decode_protected("running_mean")

        if self._original.running_var is not None:
            self._original.running_var = self._decode_protected("running_var")

        return self._original, self._failed_decodings

    def forward(self) -> None:
        raise RuntimeError(
            "Hamming layers need to be decoded before usage. See `hamming_decode_module`"
        )


def encode_module(module: nn.Module) -> None:
    """Recursively replace child layers of the module with `HammingLayer`

    A module that has been prepared like this can be used as an input for
    `hamming_layer_fi` for fault injection.

    Use `hamming_decode_module` to restore the original representation.

    See `SupportsHamming` for supported layer types.
    """
    for name, child in module.named_children():
        encode_module(child)

        if not isinstance(child, SupportsHamming):
            continue

        setattr(module, name, HammingLayer(child))


def decode_module(module: nn.Module):
    """Decodes all `HammingLayer` children into their original instances.

    This corrects all single bit errors in a memory line caused by `hamming_layer_fi`.

    See `hamming_encode_module`.
    """
    for name, child in module.named_children():
        decode_module(child)

        if not isinstance(child, HammingLayer):
            continue

        result = child.decode()

        setattr(module, name, result[0])


def protected_fi(
    module: nn.Module,
    bit_error_rate: float,
    stats: HammingStats | None = None,
) -> None:
    """Inject faults uniformly into `HammingLayer` children of the module.

    All bit flips will be unique.

    See `hamming_encode_module` to prepare the input.
    See `non_protected_fi` for the unprotected variant.
    """
    buffers = list(
        x[1]
        for x in module.named_buffers(recurse=True, remove_duplicate=False)
        if DATA_PREFIX in x[0]
    )

    tensor_list_fi(buffers, bit_error_rate, stats=stats)


def collect_supports_hamming_tensors(module: nn.Module) -> list[torch.Tensor]:
    out = []
    for child in module.children():
        out += collect_supports_hamming_tensors(child)

    if not isinstance(module, SupportsHamming):
        return out

    out.append(module.weight.data)

    if module.bias is not None:
        out.append(module.bias.data)

    if not isinstance(module, nn.BatchNorm2d):
        return out

    if module.running_mean is not None:
        out.append(module.running_mean.data)

    if module.running_var is not None:
        out.append(module.running_var.data)

    return out


def nonprotected_fi(
    module: nn.Module,
    bit_error_rate: float,
    *,
    stats: HammingStats | None = None,
) -> None:
    """Uniformly inject faults into all modules which could be encoded as HammingLayers."""
    buffers = collect_supports_hamming_tensors(module)
    tensor_list_fi(buffers, bit_error_rate, stats=stats)
