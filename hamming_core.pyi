import numpy as np
import numpy.typing as npt

__all__ = [
    "compare_array_list_bitwise_f32",
    "compare_array_list_bitwise_u16",
    "f32_array_list_fi",
    "u16_array_list_fi",
]

def f32_array_list_fi(
    input: list[npt.NDArray[np.float32]], faults_count: int
) -> list[npt.NDArray[np.float32]]: ...
def u16_array_list_fi(
    input: list[npt.NDArray[np.uint16]], faults_count: int
) -> list[npt.NDArray[np.uint16]]: ...
def compare_array_list_bitwise_f32(
    a: list[npt.NDArray[np.float32]],
    b: list[npt.NDArray[np.float32]],
) -> list[int]: ...
def compare_array_list_bitwise_u16(
    a: list[npt.NDArray[np.float32]],
    b: list[npt.NDArray[np.float32]],
) -> list[int]: ...
