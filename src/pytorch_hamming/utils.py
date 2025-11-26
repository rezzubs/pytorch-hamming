import torch


def dtype_bits_count(dtype: torch.dtype) -> int:
    """Return the number of bits for a PyTorch data type."""
    match dtype:
        case torch.float64 | torch.uint64 | torch.int64:
            return 32
        case torch.float32 | torch.uint32 | torch.int32:
            return 32
        case torch.float16 | torch.uint16 | torch.int16:
            return 16
        case torch.uint8 | torch.int8:
            return 8
        case _:
            raise ValueError(f"Unsupported datatype {dtype}")
