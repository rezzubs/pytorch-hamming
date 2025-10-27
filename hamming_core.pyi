from typing import TypeAlias, TypeVar
import numpy as np
import numpy.typing as npt

__all__ = [
    "compare_array_list_bitwise_f32",
    "compare_array_list_bitwise_u16",
    "f32_array_list_fi",
    "u16_array_list_fi",
]

T = TypeVar("T", bound=np.generic)
ListOfArray: TypeAlias = list[npt.NDArray[T]]

def f32_array_list_fi(
    input: ListOfArray[np.float32],
    faults_count: int,
) -> ListOfArray[np.float32]: ...
def u16_array_list_fi(
    input: ListOfArray[np.uint16],
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
    encoded: ListOfArray[np.uint8],
    encoded_bits_count: int,
    bits_per_chunk: int,
    decoded_array_element_counts: list[int],
) -> tuple[ListOfArray[np.float32], list[bool]]: ...
def decode_full_u16(
    encoded: ListOfArray[np.uint8],
    encoded_bits_count: int,
    bits_per_chunk: int,
    decoded_array_element_counts: list[int],
) -> tuple[ListOfArray[np.uint16], list[bool]]: ...
