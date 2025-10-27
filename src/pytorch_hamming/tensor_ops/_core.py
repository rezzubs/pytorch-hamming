import logging

import hamming_core
import torch

from pytorch_hamming.utils import dtype_num_bits

from .._dtype import DnnDtype, FiDtype


class DtypeError(Exception):
    """An error related to tensor data types."""


logger = logging.getLogger(__name__)


def tensor_list_dtype(ts: list[torch.Tensor]) -> torch.dtype | None:
    """Confirms that all tensors in `ts` have the same datatype.

    Returns:
        The common dtype of `ts` or None if `ts` is empty.

    Raises:
        DtypeMismatchError:
            If dtype values don't match.
    """
    dtype = None

    for i, t in enumerate(ts):
        if dtype is None:
            dtype = t.dtype
        else:
            if dtype != t.dtype:
                raise DtypeError(
                    f"dtype=`{t.dtype}` for tensor {i} while all previous values had dtype=`{dtype}`"
                )
    return dtype


def tensor_list_num_bits(ts: list[torch.Tensor]) -> int:
    """Get the total number of bits in a list of tensors."""
    total = 0

    for t in ts:
        total += dtype_num_bits(t.dtype) * t.numel()

    return total


def tensor_list_fault_injection(
    ts: list[torch.Tensor], num_faults: int, bit_limit: int | None = None
):
    """Flip `num_faults` unique bits in `ts`.

    Raises:
        DtypeMismatchError:
            If values in `ts` don't all have the same data type.
        ValueError:
            - If num_faults is greated than the number of bits `ts`.
            - If the data type is unsupported, see `FiDtype`.
    """
    dtype = tensor_list_dtype(ts)

    if dtype is None:
        logger.warning("Skipping fault injection because the input buffer is empty")
        return

    # NOTE: the length checks are handled in rust.
    match FiDtype.from_torch(dtype):
        case FiDtype.Float32:
            # NOTE: the tensors need to be on the CPU to use `.numpy` without `force=True`.
            flattened = [t.flatten() for t in ts]
            devices = [t.device for t in ts]

            with torch.no_grad():
                input = [t.cpu().numpy() for t in flattened]

                result = hamming_core.f32_array_list_fi(input, num_faults, bit_limit)
                torch_result = [
                    # HACK: There's nothing we can do about this warning without an upstream fix.
                    torch.from_numpy(t).to(device)  # pyright: ignore[reportUnknownMemberType]
                    for t, device in zip(result, devices, strict=True)
                ]

                for original, updated in zip(flattened, torch_result, strict=True):
                    _ = original.copy_(updated)
        case FiDtype.Float16:
            # NOTE: the tensors need to be on the CPU to use `.numpy` without `force=True`.
            flattened = [t.flatten() for t in ts]
            devices = [t.device for t in ts]

            with torch.no_grad():
                input = [t.cpu().view(torch.uint16).numpy() for t in flattened]

                result = hamming_core.u16_array_list_fi(input, num_faults, bit_limit)
                torch_result = [
                    # HACK: There's nothing we can do about this warning without an upstream fix.
                    torch.from_numpy(t).view(torch.float16).to(device)  # pyright: ignore[reportUnknownMemberType]
                    for t, device in zip(result, devices, strict=True)
                ]

                for original, updated in zip(flattened, torch_result, strict=True):
                    _ = original.copy_(updated)
        case FiDtype.Uint8:
            raise NotImplementedError


def tensor_list_compare_bitwise(
    left: list[torch.Tensor], right: list[torch.Tensor]
) -> list[int]:
    left_dtype = tensor_list_dtype(left)
    right_dtype = tensor_list_dtype(right)

    if left_dtype != right_dtype:
        raise DtypeError(
            f"The data types of the two lists don't match {left_dtype}!={right_dtype}"
        )

    if left_dtype is None:
        logger.warning("compared empty tensor lists the result will be empty")
        return []

    match DnnDtype.from_torch(left_dtype):
        case DnnDtype.Float32:
            left_numpy = [t.flatten().numpy(force=True) for t in left]
            right_numpy = [t.flatten().numpy(force=True) for t in right]
            return hamming_core.compare_array_list_bitwise_f32(left_numpy, right_numpy)
        case DnnDtype.Float16:
            #  A view is like a bitwise transmute. This is
            # necessary because rust's `f16` isn't yet stable. See:
            # https://github.com/rust-lang/rust/issues/116909
            left_numpy = [t.view(torch.uint16).flatten().numpy() for t in left]
            right_numpy = [t.view(torch.uint16).flatten().numpy() for t in right]
            return hamming_core.compare_array_list_bitwise_u16(left_numpy, right_numpy)
