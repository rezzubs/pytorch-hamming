"""Functions for working with hamming codes with 256 data bits.

The encoding functions add padding to the tensors to make the number of bits
a multiple of 256. These extra values will not be used as targets for fault
injection but cannot be removed before decoding.

The padding needs to be removed manually after decoding because there's no
way for the decoding function to know how many items were stored in the tensor
initially there's no way for the decoding function to know how many items were
stored in the tensor initially.

The encoding and decoding functions do not touch the original tensors and return
new ones instead. The fault injection function replaces the original tensor
in place.

The encoding functions will return a single dimensional tensor of bytes. The
decoding functions expect the same format as input and also return flattened
tensors because there's no way to know the original shape.
tensors instead.
"""

from ._core import (
    decode_f16,
    decode_f32,
    encode_f16,
    encode_f32,
    encoded_tensor_list_fi,
)

__all__ = [
    "decode_f16",
    "decode_f32",
    "encode_f16",
    "encode_f32",
    "encoded_tensor_list_fi",
]
