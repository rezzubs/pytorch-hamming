from hamming_utils._common import decode_impl, encode_impl, tensor_list_fi_impl
import torch
import hamming_core


# FIXME: Remove `type: ignore` comments after the hamming_core functions get
# type hints.


def encode_f32(tensor: torch.Tensor) -> torch.Tensor:
    """Enocde a flattened float32 tensor as 5 byte hamming codes.

    See module docs for details.
    """
    f = hamming_core.u32.encode_f32  # type: ignore
    return encode_impl(tensor, torch.float32, None, f)


def decode_f32(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Decode the output of `encode_f32`.

    See module docs for details.
    """
    f = hamming_core.u32.decode_f32  # type: ignore
    return decode_impl(tensor, torch.float32, None, f)


def encode_f16(tensor: torch.Tensor) -> torch.Tensor:
    """Enocde a flattened float32 tensor as 5 byte hamming codes.

    See module docs for details.
    """
    f = hamming_core.u32.encode_f16  # type: ignore
    return encode_impl(tensor, torch.float16, torch.uint16, f)


def decode_f16(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Decode the output of `encode_f16`.

    See module docs for details.
    """
    f = hamming_core.u32.decode_f16  # type: ignore
    return decode_impl(tensor, torch.float16, torch.uint16, f)


def encoded_tensor_list_fi(
    tensors: list[torch.Tensor],
    bit_error_rate: float,
    context: dict[str, int],
) -> None:
    """Inject faults uniformly in a list of tensors by the given bit error rate.

    See module docs for details.
    """
    f = hamming_core.u32.array_list_fi  # type: ignore
    tensor_list_fi_impl(
        tensors,
        bit_error_rate,
        torch.uint8,
        None,
        f,
        context,
    )
