import torch
import hamming_core

from hamming_utils._common import tensor_fi_impl, tensor_list_dtype, tensor_list_fi_impl


def f32_tensor_fi(
    tensor: torch.Tensor, bit_error_rate: float, context: dict[str, int] | None = None
) -> torch.Tensor:
    """Inject faults uniformly into the tensor by the given bit_error_rate.

    See module docs for details.
    """
    f = hamming_core.generic.f32_array_fi  # type: ignore
    return tensor_fi_impl(tensor, bit_error_rate, torch.float32, None, f, context)


def f32_tensor_list_fi(
    tensors: list[torch.Tensor],
    bit_error_rate: float,
    context: dict[str, int] | None = None,
) -> list[torch.Tensor]:
    """Inject faults uniformly into the list of tensors by the given bit_error_rate.

    See module docs for details.
    """
    f = hamming_core.generic.f32_array_list_fi  # type: ignore
    return tensor_list_fi_impl(tensors, bit_error_rate, torch.float32, None, f, context)


def f16_tensor_fi(
    tensor: torch.Tensor, bit_error_rate: float, context: dict[str, int] | None = None
) -> torch.Tensor:
    """Inject faults uniformly into the tensor by the given bit_error_rate.

    See module docs for details.
    """
    f = hamming_core.generic.u16_array_fi  # type: ignore
    return tensor_fi_impl(
        tensor, bit_error_rate, torch.float16, torch.uint16, f, context
    )


def f16_tensor_list_fi(
    tensors: list[torch.Tensor],
    bit_error_rate: float,
    context: dict[str, int] | None = None,
) -> list[torch.Tensor]:
    """Inject faults uniformly into the list of tensors by the given bit_error_rate.

    See module docs for details.
    """
    f = hamming_core.generic.u16_array_list_fi  # type: ignore
    return tensor_list_fi_impl(
        tensors, bit_error_rate, torch.float16, torch.uint16, f, context
    )


def tensor_fi(
    tensor: torch.Tensor, bit_error_rate: float, context: dict[str, int] | None = None
) -> torch.Tensor:
    """Inject faults uniformly into the tensor by the given bit_error_rate.

    See module docs for details.
    """

    if tensor.dtype == torch.float32:
        return f32_tensor_fi(tensor, bit_error_rate, context)
    elif tensor.dtype == torch.float16:
        return f16_tensor_fi(tensor, bit_error_rate, context)
    else:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")


def tensor_list_fi(
    tensors: list[torch.Tensor],
    bit_error_rate: float,
    context: dict[str, int] | None = None,
) -> list[torch.Tensor]:
    """Inject faults uniformly into the list of tensors by the given bit_error_rate.

    See module docs for details.
    """
    dtype = tensor_list_dtype(tensors)
    if dtype == torch.float32:
        return f32_tensor_list_fi(tensors, bit_error_rate, context)
    elif dtype == torch.float16:
        return f16_tensor_list_fi(tensors, bit_error_rate, context)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
