"""Utilities for encoding PyTorch modules with hamming codes."""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from typing import Any

import numpy
import torch
from torch import nn

import hamming_core

__all__ = [
    "HammingLayer",
    "HammingStats",
    "decode_f32",
    "decode_module",
    "encode_f32",
    "encode_module",
    "protected_fi",
    "nonprotected_fi",
]

HAMMING_DATA_PREFIX = "hamming_protected_"
BITS_PER_BYTE = 8
BYTES_PER_CONTAINER = 9
BITS_PER_CONTAINER = BITS_PER_BYTE * BYTES_PER_CONTAINER


def num_set_bits32(number: int) -> int:
    """Count the high bits in an integer

    `number` is expected to be a 32 bit integer.
    """
    # Force an unsigned 32 bit representation. Otherwise -1 >> 1 will cause an
    # infinite loop.
    number = number & 0xFFFFFFFF
    count = 0
    while number != 0:
        if number & 1:
            count += 1
        number >>= 1

    return count


class HammingStats:
    """Statistics for a encode, inject, decode cycle."""

    def __init__(self) -> None:
        self.was_protected: bool = False
        self.accuracy: float | None = None
        self.injected_faults: list[int] = []
        self.total_bits: int | None = None
        self.unsuccessful_corrections: int | None = None
        # The xors between all true and faulty parameters.
        self.non_matching_parameters: list[int] = []

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, HammingStats):
            raise ValueError(f"Can't compare HammingStats with {type(value)}")
        return (
            self.was_protected == value.was_protected
            and self.accuracy == value.accuracy
            and self.injected_faults == value.injected_faults
            and self.total_bits == value.total_bits
            and self.unsuccessful_corrections == value.unsuccessful_corrections
            and self.non_matching_parameters == value.non_matching_parameters
        )

    @classmethod
    def eval(
        cls,
        module: nn.Module,
        bit_error_rate: float,
        accuracy_fn: Callable[[nn.Module], float],
    ) -> HammingStats:
        stats = cls()
        original = copy.deepcopy(module)

        encode_module(module)
        protected_fi(module, bit_error_rate=bit_error_rate, stats=stats)
        decode_module(module, stats=stats)

        stats.accuracy = accuracy_fn(module)
        stats.non_matching_parameters = compare_module_bitwise(module, original)

        return stats

    @classmethod
    def eval_noprotect(
        cls,
        module: nn.Module,
        bit_error_rate: float,
        accuracy_fn: Callable[[nn.Module], float],
    ) -> HammingStats:
        stats = cls()
        original = copy.deepcopy(module)

        nonprotected_fi(module, bit_error_rate=bit_error_rate, stats=stats)

        stats.accuracy = accuracy_fn(module)
        stats.non_matching_parameters = compare_module_bitwise(module, original)

        return stats

    def faulty_containers(self) -> dict[int, list[int]]:
        output: dict[int, list[int]] = dict()

        for bit in self.injected_faults:
            container_idx = bit // BITS_PER_CONTAINER
            bit_idx = bit % BITS_PER_CONTAINER

            faults_per_container = output.get(container_idx, [])
            faults_per_container.append(bit_idx)
            output[container_idx] = faults_per_container

        return output

    def get_accuracy(self) -> float:
        assert self.accuracy is not None
        return self.accuracy

    def n_flips_per_param(self) -> dict[int, int]:
        """Return the number of parameters grouped by the number of bits flipped in each."""

        out = dict()
        for p in self.non_matching_parameters:
            group = num_set_bits32(p)
            assert group != 0
            prev = out.get(group, 0)
            out[group] = prev + 1

        return out

    def num_faults(self) -> int:
        return len(self.injected_faults)

    def num_faults_in_result(self) -> int:
        return sum([num_faulty for num_faulty in self.n_flips_per_param().values()])

    def output_bit_error_rate(self) -> float:
        assert self.total_bits is not None
        return self.num_faults_in_result() / self.total_bits

    def protection_rate(self) -> float:
        return 1 - (
            self.num_faults_in_result() / self.num_faults()
            if self.num_faults() > 0
            else 0
        )

    def bit_error_rate(self) -> float:
        assert self.total_bits is not None
        return len(self.injected_faults) / self.total_bits

    def summary(self) -> None:
        print("Fault Injection Summary:")
        num_faults = len(self.injected_faults)
        assert self.total_bits is not None
        print(
            f"  Injected {num_faults} faults across {self.total_bits} bits, BER: {num_faults / self.total_bits}"
        )
        print(f"  Accuracy: {self.accuracy:.2f}%")

        if self.was_protected:
            faulty = self.faulty_containers()
            exactly_one = len([v for v in faulty.values() if len(v) == 1])
            exactly_two = len([v for v in faulty.values() if len(v) == 2])
            three_or_more = len([v for v in faulty.values() if len(v) >= 3])
            assert self.unsuccessful_corrections is not None

            print(f"  {exactly_one} containers had exactly 1 fault")
            print(f"  {exactly_two} containers had exactly 2 faults")
            print(f"  {three_or_more} containers had 3 or more faults")
            print(
                f"  Decoding detected {self.unsuccessful_corrections} non-correctable containers (an even number of faults or bit 0)"
            )

        print(
            f"  {len(self.non_matching_parameters)} parameters were messed up from injection"
        )
        param_fault_groups = list(self.n_flips_per_param().items())
        param_fault_groups.sort(key=lambda x: x[0])
        for num_faults, num_entries in param_fault_groups:
            print(f"  {num_entries} parameters had {num_faults} faults")

    def to_dict(self) -> dict:
        out = dict()

        if self.accuracy is None:
            raise ValueError("Incomplete stats, accuracy missing, run inference")

        out["accuracy"] = self.accuracy
        out["injected_faults"] = self.injected_faults

        if self.total_bits is None:
            raise ValueError(
                "Incomplete stats, injected_faults missing, run fault injection"
            )

        out["total_bits"] = self.total_bits

        out["was_protected"] = self.was_protected
        if self.was_protected:
            assert self.unsuccessful_corrections is not None
            out["unsuccessful_corrections"] = self.unsuccessful_corrections
        else:
            assert self.unsuccessful_corrections is None

        out["non_matching_parameters"] = self.non_matching_parameters

        return out

    @classmethod
    def from_dict(cls, obj: Any) -> HammingStats:
        out = cls()

        if not isinstance(obj, dict):
            raise ValueError(f"Expected a dictionary, got {type(obj)}")

        accuracy = obj["accuracy"]
        if not isinstance(accuracy, float):
            raise ValueError(f"Expected float, got {type(accuracy)}")
        out.accuracy = accuracy

        injected_faults = obj["injected_faults"]
        if not isinstance(injected_faults, list):
            raise ValueError(f"Expected a list, got {type(injected_faults)}")
        for x in injected_faults:
            if not isinstance(x, int):
                raise ValueError(f"Expected int, got {type(x)}")
        out.injected_faults = injected_faults

        total_bits = obj["total_bits"]
        if not isinstance(total_bits, int):
            raise ValueError(f"Expected int, got {type(total_bits)}")
        out.total_bits = total_bits

        was_protected = obj["was_protected"]
        if not isinstance(was_protected, bool):
            raise ValueError(f"Expected bool, got {type(was_protected)}")
        out.was_protected = was_protected

        if was_protected:
            unsuccessful_corrections = obj["unsuccessful_corrections"]
            if not isinstance(unsuccessful_corrections, int):
                raise ValueError(f"Expected int, got {type(unsuccessful_corrections)}")
            out.unsuccessful_corrections = unsuccessful_corrections

        non_matching_parameters = obj["non_matching_parameters"]
        if not isinstance(non_matching_parameters, list):
            raise ValueError(f"Expected a list, got {type(non_matching_parameters)}")
        for x in non_matching_parameters:
            if not isinstance(x, int):
                raise ValueError(f"Expected int, got {type(x)}")
        out.non_matching_parameters = non_matching_parameters

        return out


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
    """Decode the output of `hamming_encode_f32`.

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


SupportsHamming = nn.Linear | nn.Conv2d | nn.BatchNorm2d


def compare_parameter_bitwise(a: torch.Tensor, b: torch.Tensor) -> list[int]:
    assert a.shape == b.shape
    assert a.dtype == b.dtype == torch.float32

    out = []
    for a_item, b_item in zip(a.flatten(), b.flatten(), strict=True):
        a_bits = int(a_item.view(torch.int32).item())
        b_bits = int(b_item.view(torch.int32).item())

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
        og = "hamming_original_" + name
        t = t.data

        protected_data = encode_f32(t.flatten())
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

        result = decode_f32(protected_data)
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


def decode_module(module: nn.Module, *, stats: HammingStats | None = None):
    """Decodes all `HammingLayer` children into their original instances.

    This corrects all single bit errors in a memory line caused by `hamming_layer_fi`.

    See `hamming_encode_module`.
    """
    if stats is not None:
        stats.was_protected = True

    for name, child in module.named_children():
        decode_module(child, stats=stats)

        if not isinstance(child, HammingLayer):
            continue

        result = child.decode()
        if stats is not None:
            if stats.unsuccessful_corrections is None:
                stats.unsuccessful_corrections = result[1]
            else:
                stats.unsuccessful_corrections += result[1]

        setattr(module, name, result[0])


def tensor_list_dtype(buffers: list[torch.Tensor]) -> torch.dtype:
    if len(buffers) == 0:
        raise ValueError("Expected at least 1 element")

    dtype = buffers[0].dtype
    for buf in buffers[1:]:
        if buf.dtype != dtype:
            raise ValueError(
                f"Expected all tensors to have the same datatype, got {dtype} and {buf.dtype}"
            )

    return dtype


def bits_per_dtype(dtype: torch.dtype) -> int:
    if dtype == torch.uint8:
        return 8
    elif dtype == torch.float32:
        return 32
    else:
        raise ValueError(f"Unsupported datatype {dtype}")


def uint8_tensor_flip_bit(t: torch.Tensor, bit_index: int) -> None:
    """Flip a single bit in a uint8 tensor.

    The values in the tensor are treated as a continuous stream of bits.
    """
    if t.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 tensor, got {t.dtype}")
    if len(t.shape) != 1:
        raise ValueError(f"Expected a single dimensional tensor, got shape {t.shape}")

    dtype_bits = bits_per_dtype(torch.uint8)

    num_bits = t.numel() * dtype_bits

    if bit_index >= num_bits:
        raise ValueError(f"Tensor has {num_bits} bits, got index {bit_index}")

    byte_index = bit_index // dtype_bits
    true_bit_index = bit_index % dtype_bits

    t[byte_index] = t[byte_index] ^ (1 << true_bit_index)


def float32_tensor_flip_bit(t: torch.Tensor, bit_index: int) -> None:
    if t.dtype != torch.float32:
        raise ValueError(f"Expected float32 tensor, got {t.dtype}")
    if len(t.shape) != 1:
        raise ValueError(f"Expected a single dimensional tensor, got shape {t.shape}")

    dtype_bits = bits_per_dtype(torch.float32)

    num_bits = t.numel() * dtype_bits

    if bit_index >= num_bits:
        raise ValueError(f"Tensor has {num_bits} bits, got index {bit_index}")

    byte_index = bit_index // dtype_bits
    true_bit_index = bit_index % dtype_bits

    bits = t[byte_index].view(torch.int32)
    faulty = (bits ^ (1 << true_bit_index)).view(torch.float32)

    t[byte_index] = faulty


def tensor_flip_bit(t: torch.Tensor, bit_index: int) -> None:
    if t.dtype == torch.uint8:
        uint8_tensor_flip_bit(t, bit_index)
    elif t.dtype == torch.float32:
        float32_tensor_flip_bit(t, bit_index)


def tensor_list_flip_bit(ts: list[torch.Tensor], bit_index: int) -> None:
    """Flip a single bit in a list of tensors.

    The list of tensors are interpreted as a continuous stream of bits.
    """
    dtype = tensor_list_dtype(ts)
    dtype_bits = bits_per_dtype(dtype)

    start_bit = 0
    for t in ts:
        num_bits = t.numel() * dtype_bits
        first_bit_of_next = start_bit + num_bits

        if first_bit_of_next <= bit_index:
            start_bit = first_bit_of_next
            continue

        t_bit_index = bit_index - start_bit
        assert t_bit_index >= 0

        tensor_flip_bit(t, t_bit_index)
        return

    total_num_bits = sum([t.numel() * dtype_bits for t in ts])
    raise ValueError(
        f"Tensor list has {total_num_bits} bits in total, got index {bit_index}"
    )


def buffers_fi(
    buffers: list[torch.Tensor],
    *,
    num_faults: int = 0,
    bit_error_rate: float | None = None,
    stats: HammingStats | None = None,
) -> None:
    """Perform fault injection on a series of tensors.

    These tensors will be treated as one large buffer and must have the same datatype.

    Supported datatypes are uint8 and float32.
    """
    dtype = tensor_list_dtype(buffers)

    total_num_bits = sum([t.numel() * bits_per_dtype(dtype) for t in buffers])

    if stats is not None:
        stats.total_bits = total_num_bits

    if total_num_bits < num_faults:
        raise ValueError(
            f"The module has {total_num_bits} bits worth of unprotected data, "
            "tried to inject {num_flips} faults"
        )

    if bit_error_rate is not None:
        if not (0.0 <= bit_error_rate <= 1.0):
            raise ValueError(
                f"`bit_error_rate` must be between 0 and 1, got {bit_error_rate}"
            )
        num_faults = int(round(bit_error_rate * total_num_bits))

    flip_candidates = list(range(total_num_bits))
    random.shuffle(flip_candidates)

    for _ in range(num_faults):
        bit_to_flip = flip_candidates.pop()
        if stats is not None:
            stats.injected_faults.append(bit_to_flip)

        tensor_list_flip_bit(buffers, bit_to_flip)


def protected_fi(
    module: nn.Module,
    *,
    num_faults: int = 0,
    bit_error_rate: float | None = None,
    stats: HammingStats | None = None,
) -> None:
    """Inject faults uniformly into `HammingLayer` children of the module.

    All bit flips will be unique.

    Args:
        num_flips: How many bits to flip.
        bit_error_rate:
            Compute `num_bits` as a percentage of the total number of bits.
            Overrides `num_flips`.

    See `hamming_encode_module` to prepare the input.
    See `supports_hamming_fi` for the unprotected variant.
    """
    protected_buffers = list(
        x[1]
        for x in module.named_buffers(recurse=True, remove_duplicate=False)
        if HAMMING_DATA_PREFIX in x[0]
    )

    buffers_fi(
        protected_buffers,
        num_faults=num_faults,
        bit_error_rate=bit_error_rate,
        stats=stats,
    )


def collect_supports_hamming_tensors(module: nn.Module) -> list[torch.Tensor]:
    out = []
    for child in module.children():
        out += collect_supports_hamming_tensors(child)

    if not isinstance(module, SupportsHamming):
        return out

    out.append(module.weight.data.flatten())

    if module.bias is not None:
        out.append(module.bias.data.flatten())

    if not isinstance(module, nn.BatchNorm2d):
        return out

    if module.running_mean is not None:
        out.append(module.running_mean.data.flatten())

    if module.running_var is not None:
        out.append(module.running_var.data.flatten())

    return out


def nonprotected_fi(
    module: nn.Module,
    *,
    num_faults: int = 0,
    bit_error_rate: float | None = None,
    stats: HammingStats | None = None,
) -> None:
    """Uniformly inject faults into all modules which could be encoded as HammingLayers."""
    buffers = collect_supports_hamming_tensors(module)

    buffers_fi(
        buffers,
        num_faults=num_faults,
        bit_error_rate=bit_error_rate,
        stats=stats,
    )
