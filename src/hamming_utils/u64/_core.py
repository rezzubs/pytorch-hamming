from hamming_utils._common import tensor_list_fi_impl
import torch
import hamming_core
import numpy as np


def encode_f32(t: torch.Tensor) -> torch.Tensor:
    """Enocde a flattened float32 tensor as 9 byte hamming codes.

    See module docs for details.
    """
    if len(t.shape) != 1:
        raise ValueError(f"Expected a flattened tensor, got shape {t.shape}")

    # TODO: match on dtype and add support for f16-f64.
    if t.dtype != torch.float32:
        raise ValueError(f"Only float32 tensors are supported, got {t.dtype}")

    # FIXME: Ignored because there are no type signatures for the hamming module.
    out: np.ndarray = hamming_core.u64.encode_f32(t.numpy())  # pyright: ignore

    return torch.from_numpy(out)


def decode_f32(t: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Decode the output of `encode_f32`.

    See module docs for details.
    """
    if len(t.shape) != 1:
        raise ValueError(f"Expected a flattened tensor, got shape {t.shape}")

    if t.dtype != torch.uint8:
        raise ValueError(f"Expected dtype=uint8, got {t.dtype}")

    # NOTE: Length checks are handled in rust.
    # FIXME: Ignored because there are no type signatures for the hamming module.
    result: tuple[np.ndarray, int] = hamming_core.u64.decode_f32(t.numpy())  # pyright: ignore

    return torch.from_numpy(result[0]), result[1]


def encode_f16(t: torch.Tensor) -> torch.Tensor:
    """Enocde a flattened float32 tensor as 9 byte hamming codes.

    See module docs for details.
    """
    if len(t.shape) != 1:
        raise ValueError(f"Expected a flattened tensor, got shape {t.shape}")

    if t.dtype != torch.float16:
        raise ValueError(f"Expected dtype=float16, got {t.dtype}")

    result: np.ndarray = hamming_core.u64.encode_u16(t.view(torch.uint16).numpy())  # pyright: ignore
    torch_result = torch.from_numpy(result)
    assert torch_result.dtype == torch.uint8

    return torch_result


def decode_f16(t: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Decode the output of `encode_f16`.

    See module docs for details.
    """
    if len(t.shape) != 1:
        raise ValueError(f"Expected a flattened tensor, got shape {t.shape}")

    if t.dtype != torch.uint8:
        raise ValueError(f"Expected dtype=uint8, got {t.dtype}")

    # NOTE: Length checks are handled in rust.
    # FIXME: Ignored because there are no type signatures for the hamming module.
    result: tuple[np.ndarray, int] = hamming_core.u64.decode_u16(t.numpy())  # pyright: ignore
    torch_result = torch.from_numpy(result[0])
    assert torch_result.dtype == torch.uint16

    return torch_result.view(torch.float16), result[1]


def encoded_tensor_list_fi(
    tensors: list[torch.Tensor],
    bit_error_rate: float,
    context: dict[str, int],
) -> None:
    """Inject faults uniformly in a list of tensors by the given bit error rate.

    See module docs for details.
    """
    f = hamming_core.u64.array_list_fi  # type: ignore
    tensor_list_fi_impl(
        tensors,
        bit_error_rate,
        torch.uint8,
        None,
        f,
        context,
    )
