import torch
import numpy as np
from collections.abc import Callable
from typing import Any


def tensor_fi_impl(
    tensor: torch.Tensor,
    bit_error_rate: float,
    dtype: torch.dtype,
    view: torch.dtype | None,
    rust_fn: Callable[[np.ndarray, float], Any],
    context: dict[str, int] | None,
) -> torch.Tensor:
    if tensor.dtype != dtype:
        raise ValueError(f"Expected dtype={dtype}, got {tensor.dtype}")

    if not (0 <= bit_error_rate <= 1):
        raise ValueError("Bit error rate is expected to be within 0 and 1 inclusive")

    flattened = tensor.flatten()
    input = flattened if view is None else flattened.view(view)

    result, context_ = rust_fn(input.numpy(), bit_error_rate)
    assert isinstance(result, np.ndarray)
    assert isinstance(context_, dict)

    result = torch.from_numpy(result)

    if view is not None:
        assert result.dtype == view
        result = result.view(dtype)
    else:
        assert result.dtype == dtype

    flattened.copy_(result)

    if context is not None:
        context.update(context_)

    return tensor


def tensor_list_fi_impl(
    tensors: list[torch.Tensor],
    bit_error_rate: float,
    dtype: torch.dtype,
    view: torch.dtype | None,
    rust_fn: Callable[[list[np.ndarray], float], Any],
    context: dict[str, int] | None,
) -> list[torch.Tensor]:
    for i, tensor in enumerate(tensors):
        if tensor.dtype != dtype:
            raise ValueError(f"Expected dtype={dtype}, got {tensor.dtype} (tensor {i})")

    if not (0 <= bit_error_rate <= 1):
        raise ValueError("Bit error rate is expected to be within 0 and 1 inclusive")

    flattened = [t.flatten() for t in tensors]
    input: list[torch.Tensor] = []
    for tensor in flattened:
        item = tensor if view is None else tensor.view(view)
        input.append(item)

    result, context_ = rust_fn([t.numpy() for t in input], bit_error_rate)
    assert isinstance(result, list)
    assert isinstance(context_, dict)

    for original, new in zip(flattened, result, strict=True):
        assert isinstance(new, np.ndarray)
        torch_item = torch.from_numpy(new)

        if view is not None:
            assert torch_item.dtype == view
            torch_item = torch_item.view(dtype)
        else:
            assert torch_item.dtype == dtype

        original.copy_(torch_item)

    if context is not None:
        context.update(context_)

    return tensors


def tensor_list_dtype(tensors: list[torch.Tensor]) -> torch.dtype | None:
    dtype = None
    for tensor in tensors:
        if dtype is None:
            dtype = tensor.dtype

        if tensor.dtype != dtype:
            raise ValueError(f"Received different dtypes {dtype} vs {tensor.dtype}")

    return dtype


def encode_impl(
    tensor: torch.Tensor,
    dtype: torch.dtype,
    view: torch.dtype | None,
    rust_fn: Callable[[np.ndarray], Any],
) -> torch.Tensor:
    if tensor.dtype != dtype:
        raise ValueError(f"Expected dtype={dtype}, got {tensor.dtype}")

    if view is not None:
        tensor = tensor.view(view)

    result = rust_fn(tensor.flatten().numpy())
    assert isinstance(result, np.ndarray)

    torch_result = torch.from_numpy(result)
    assert torch_result.dtype == torch.uint8

    return torch_result


def decode_impl(
    tensor: torch.Tensor,
    dtype: torch.dtype,
    view: torch.dtype | None,
    rust_fn: Callable[[np.ndarray], Any],
) -> tuple[torch.Tensor, int]:
    if tensor.dtype != torch.uint8:
        raise ValueError(f"Expected dtype=uint8, got {tensor.dtype}")

    if len(tensor.shape) != 1:
        raise ValueError(f"Expected a single dimensional tensor, got {tensor.shape}")

    result, num_failures = rust_fn(tensor.numpy())
    assert isinstance(result, np.ndarray)
    assert isinstance(num_failures, int)

    torch_result = torch.from_numpy(result)

    if view is not None:
        assert torch_result.dtype == view
        torch_result = torch_result.view(dtype)
    else:
        assert torch_result.dtype == dtype

    return torch_result, num_failures
