from typing import NoReturn
import torch


def dtype_num_bits(dtype: torch.dtype) -> int:
    """Return the number of bits for a PyTorch data type."""
    match dtype:
        case torch.float64 | torch.uint64 | torch.int64:
            return 32
        case torch.float32 | torch.uint32 | torch.int32:
            return 32
        case torch.float16 | torch.uint16 | torch.int16:
            return 16
        case _:
            raise ValueError(f"Unsupported datatype {dtype}")


def unreachable(*args: object) -> NoReturn:
    raise RuntimeError("Unreachable", args)
