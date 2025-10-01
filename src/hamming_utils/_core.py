"""Utilities for encoding PyTorch modules with hamming codes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import hamming_core

import torch
from torch import nn

from . import generic
from . import u32 as u32_impl
from . import u64 as u64_impl
from . import u128 as u128_impl
from . import u256 as u256_impl


if TYPE_CHECKING:
    from ._stats import HammingStats

DATA_PREFIX = "hamming_protected_"
ORIGINAL_PREFIX = "hamming_original_"
DATA_PREFIX = "hamming_protected_"

DTYPE_F32 = 0
DTYPE_F16 = 1

DATA_BUFFER_U32 = 32
DATA_BUFFER_U64 = 64
DATA_BUFFER_U128 = 128
DATA_BUFFER_U256 = 256


SupportsHamming = nn.Linear | nn.Conv2d | nn.BatchNorm2d


class HammingImpl(Protocol):
    def encode_f32(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def decode_f32(self, tensor: torch.Tensor) -> tuple[torch.Tensor, int]: ...

    def encode_f16(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def decode_f16(self, tensor: torch.Tensor) -> tuple[torch.Tensor, int]: ...

    def encoded_tensor_list_fi(
        self,
        tensors: list[torch.Tensor],
        bit_error_rate: float,
        context: dict[str, int],
    ) -> None: ...


DATA_BUFFER_SIZES: dict[int, HammingImpl] = {
    DATA_BUFFER_U32: u32_impl,
    DATA_BUFFER_U64: u64_impl,
    DATA_BUFFER_U128: u128_impl,
    DATA_BUFFER_U256: u256_impl,
}


def compare_module_bitwise(a: nn.Module, b: nn.Module) -> list[int]:
    a_params = collect_supports_hamming_tensors(a)
    b_params = collect_supports_hamming_tensors(b)

    if len(a_params) != len(b_params):
        raise ValueError("`a` and `b` have a different number of parameter tensors")

    if len(a_params) == 0:
        return []

    dtype = a_params[0].dtype

    a_params_arrays = []
    b_params_arrays = []

    def prepare_tensor(tensor: torch.Tensor):
        match tensor.dtype:
            case torch.float32:
                return tensor.numpy().flatten()
            case torch.float16:
                return tensor.view(torch.uint16).numpy().flatten()
            case _:
                raise ValueError(f"Unsupported dtype {tensor.dtype}")

    for a_param, b_param in zip(a_params, b_params):
        if a_param.dtype != b_param.dtype or a_param.dtype != dtype:
            raise ValueError(
                f"Datatype mismatch, expected: {dtype}, got {a_param.dtype} and {b_param.dtype}"
            )

        a_params_arrays.append(prepare_tensor(a_param))
        b_params_arrays.append(prepare_tensor(b_param))

    match dtype:
        case torch.float32:
            impl = hamming_core.generic.compare_array_list_bitwise_f32  # type: ignore
        case torch.float16:
            impl = hamming_core.generic.compare_array_list_bitwise_u16  # type: ignore

    result = impl(a_params_arrays, b_params_arrays)

    assert isinstance(result, list)
    for x in result:
        assert isinstance(x, int)

    return result


class HammingLayer(nn.Module):
    """A wrapper for layers in a neural network which encodes the weights as hamming codes.

    Must be decoded before usage.
    """

    def __init__(
        self,
        original: SupportsHamming,
        data_buffer_size: int = DATA_BUFFER_U64,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(original, SupportsHamming):
            raise ValueError(
                f"Module {type(original)} is not a valid HammingLayer target"
            )

        if data_buffer_size not in DATA_BUFFER_SIZES:
            raise ValueError(
                f"Unsupported data buffer size {data_buffer_size}, supported: {DATA_BUFFER_SIZES}"
            )
        self.register_buffer("hamming_buffer_size", torch.tensor(data_buffer_size))

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

    def data_buffer_size(self) -> int:
        item = self.get_buffer("hamming_buffer_size").item()
        assert isinstance(item, int)
        return item

    def _hamming_impl(self) -> HammingImpl:
        buffer_size = self.data_buffer_size()

        try:
            return DATA_BUFFER_SIZES[buffer_size]
        except KeyError:
            raise RuntimeError(f"Unexpected data buffer size {buffer_size}")

    def _protect_tensor(self, name: str, t: torch.Tensor) -> None:
        """Protect a parameter by encoding it as a hamming code.

        These parameters will be used for fault injection.

        See also `_decode_protected`.
        """
        og = ORIGINAL_PREFIX + name
        t = t.data

        impl = self._hamming_impl()

        if t.dtype == torch.float32:
            dtype = DTYPE_F32
            protected_data = impl.encode_f32(t)
        elif t.dtype == torch.float16:
            dtype = DTYPE_F16
            protected_data = impl.encode_f16(t)
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

        impl = self._hamming_impl()

        if dtype == DTYPE_F32:
            result = impl.decode_f32(protected_data)
        elif dtype == DTYPE_F16:
            result = impl.decode_f16(protected_data)
        else:
            raise ValueError(f"Unexpected dtype variant `{dtype}`")

        self._failed_decodings += result[1]
        whole_buffer = result[0]
        assert len(whole_buffer) >= length

        return whole_buffer[:length].reshape(shape)

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


def encode_module(module: nn.Module, data_buffer_size: int = DATA_BUFFER_U64) -> None:
    """Recursively replace child layers of the module with `HammingLayer`

    A module that has been prepared like this can be used as an input for
    `hamming_layer_fi` for fault injection.

    Use `hamming_decode_module` to restore the original representation.

    See `SupportsHamming` for supported layer types.
    """
    for name, child in module.named_children():
        encode_module(child, data_buffer_size)

        if not isinstance(child, SupportsHamming):
            continue

        setattr(module, name, HammingLayer(child, data_buffer_size))


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


def validate_hamming_layers(module: nn.Module, size: int | None) -> HammingImpl | None:
    impl = None

    if isinstance(module, HammingLayer):
        impl = module._hamming_impl()
        if size is None:
            size = module.data_buffer_size()
        else:
            module_size = module.data_buffer_size
            if size != module_size:
                raise ValueError(
                    f"Expected all child modules to have the same data buffer size. First saw {size}, then {module_size}"
                )

    for child in module.children():
        child_impl = validate_hamming_layers(child, size)
        if impl is None:
            impl = child_impl

    return impl


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
    impl = validate_hamming_layers(module, None)

    if impl is None:
        print(
            "Warning: Didn't detect any HammingLayer instances. Skipping fault injection."
        )
        return

    buffers = list(
        x[1]
        for x in module.named_buffers(recurse=True, remove_duplicate=False)
        if DATA_PREFIX in x[0]
    )
    context = dict()

    impl.encoded_tensor_list_fi(buffers, bit_error_rate, context)

    if stats is not None:
        stats.num_faults = context["num_faults"]
        stats.total_bits = context["total_bits"]


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
    context = dict()

    generic.tensor_list_fi(buffers, bit_error_rate, context)

    if stats is not None:
        stats.num_faults = context["num_faults"]
        stats.total_bits = context["total_bits"]
