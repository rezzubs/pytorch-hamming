"""Wrappers for the rust hamming module"""

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
        self.register_buffer("hamming_protected_" + name, protected_data)

        self.register_buffer(og + "_shape", torch.tensor(t.shape))

        self.register_buffer(og + "_len", torch.tensor(t.numel()))

    def _decode_protected(self, name: str) -> torch.Tensor:
        """Decode a protected named parameter.

        See also `_protect_tensor`
        """
        og = "hamming_original_" + name

        protected_data = self.get_buffer("hamming_protected_" + name)

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


def hamming_layer_fi(module: nn.Module, bit_error_rate: float) -> None:
    """Inject faults uniformly into `HammingLayer` children of the module.

    See `hamming_encode_module`.
    """
    raise RuntimeError("Unimplemented")
