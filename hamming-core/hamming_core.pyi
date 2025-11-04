import numpy as np
import numpy.typing as npt

__all__ = [
    "ListOfArray",
    "compare_array_list_bitwise_f32",
    "compare_array_list_bitwise_u16",
    "f32_array_list_fi",
    "u16_array_list_fi",
]

type ListOfArray[T: np.generic] = list[npt.NDArray[T]]

class BitPatternEncoding: ...

def f32_array_list_fi(
    input: ListOfArray[np.float32],
    faults_count: int,
) -> ListOfArray[np.float32]: ...
def u16_array_list_fi(
    input: ListOfArray[np.uint16],
    faults_count: int,
) -> ListOfArray[np.uint16]: ...
def u8_array_list_fi(
    input: ListOfArray[np.uint8],
    faults_count: int,
) -> ListOfArray[np.uint16]: ...
def compare_array_list_bitwise_f32(
    a: ListOfArray[np.float32],
    b: ListOfArray[np.float32],
) -> list[int]: ...
def compare_array_list_bitwise_u16(
    a: ListOfArray[np.float32],
    b: ListOfArray[np.float32],
) -> list[int]: ...
def encode_full_f32(
    input: ListOfArray[np.float32],
    bits_per_chunk: int,
) -> tuple[npt.NDArray[np.uint8], int]: ...
def encode_full_u16(
    input: ListOfArray[np.uint16],
    bits_per_chunk: int,
) -> tuple[npt.NDArray[np.uint8], int]: ...
def decode_full_f32(
    encoded: npt.NDArray[np.uint8],
    encoded_bits_count: int,
    bits_per_chunk: int,
    decoded_array_element_counts: list[int],
) -> tuple[ListOfArray[np.float32], list[bool]]: ...
def decode_full_u16(
    encoded: npt.NDArray[np.uint8],
    encoded_bits_count: int,
    bits_per_chunk: int,
    decoded_array_element_counts: list[int],
) -> tuple[ListOfArray[np.uint16], list[bool]]: ...
def encode_bit_pattern_f32(
    input: ListOfArray[np.float32],
    bit_pattern_bits: list[int],
    bit_pattern_length: int,
    bits_per_chunk: int,
) -> BitPatternEncoding: ...
def encode_bit_pattern_u16(
    input: ListOfArray[np.uint16],
    bit_pattern_bits: list[int],
    bit_pattern_length: int,
    bits_per_chunk: int,
) -> BitPatternEncoding: ...
def decode_bit_pattern_f32(
    encoded: BitPatternEncoding,
    bit_pattern_bits: list[int],
    bit_pattern_length: int,
    bits_per_chunk: int,
    decoded_array_element_counts: list[int],
) -> tuple[ListOfArray[np.uint16], list[bool]]: ...
def decode_bit_pattern_u16(
    encoded: BitPatternEncoding,
    bit_pattern_bits: list[int],
    bit_pattern_length: int,
    bits_per_chunk: int,
    decoded_array_element_counts: list[int],
) -> tuple[ListOfArray[np.uint16], list[bool]]: ...
