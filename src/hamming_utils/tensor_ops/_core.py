import torch
from hamming_utils.utils import dtype_num_bits


class DtypeMismatchError(Exception):
    """A data type mismatch was detected between tensors"""


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
                raise DtypeMismatchError(
                    f"dtype=`{t.dtype}` for tensor {i} while all previous values had dtype=`{dtype}`"
                )


def tensor_list_num_bits(ts: list[torch.Tensor]) -> int:
    """Get the total number of bits in a list of tensors."""
    total = 0

    for t in ts:
        total += dtype_num_bits(t.dtype)

    return total


def tensor_list_fault_injection(ts: list[torch.Tensor], num_faults: int):
    """Flip `num_faults` unique bits in `ts`.

    Raises:
        DtypeMismatchError:
            If values in `ts` don't all have the same data type.
        ValueError:
            If num_faults is greated than the number of bits `ts`

    """
    _ = tensor_list_dtype(ts)
    _ = num_faults

    raise NotImplementedError


def tensor_list_compare_bitwise(
    left: list[torch.Tensor], right: list[torch.Tensor]
) -> list[int]:
    left_dtype = tensor_list_dtype(left)
    right_dtype = tensor_list_dtype(right)
    if left_dtype != right_dtype:
        raise ValueError(
            f"The data types of the two lists don't match {left_dtype}!={right_dtype}"
        )

    raise NotImplementedError
