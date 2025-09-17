"""Functions for fault injection on arbitrary tensors.

All bit flips will be unique and the original tensor is replaced through
the reference.

The `context` parameter will be used to save additional context during injection

Currently the supported data types include float32 and float16
"""

from ._core import (
    f16_tensor_fi,
    f16_tensor_list_fi,
    f32_tensor_fi,
    f32_tensor_list_fi,
    tensor_fi,
    tensor_list_fi,
)

__all__ = [
    "f16_tensor_fi",
    "f16_tensor_list_fi",
    "f32_tensor_fi",
    "f32_tensor_list_fi",
    "tensor_fi",
    "tensor_list_fi",
]
