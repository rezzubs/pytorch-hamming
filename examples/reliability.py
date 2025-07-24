"""Enocode and validate error detection for 10_000 attempts."""

from pytorch_ecc import hamming_encode, hamming_error_bit_index
from pytorch_ecc._utils import toggle_bit

import random
import torch


TENSOR_SIZE = 10


def main():
    print("Injecting errors into 10_000 tensors")

    for _ in range(10_000):
        starting_values = [
            random.uniform(-1_000_000.0, 1_000_000.0) for _ in range(TENSOR_SIZE)
        ]
        initial = hamming_encode(torch.tensor(starting_values))

        toggled_bit = random.randint(0, 38)  # 38 is the index of the final data bit.
        faulty = toggle_bit(initial, toggled_bit)

        error = hamming_error_bit_index(faulty)
        expected_error = torch.tensor(
            [toggled_bit for _ in range(TENSOR_SIZE)], dtype=error.dtype
        )

        assert torch.equal(error, expected_error)

    print("Caught all errors successfully")


if __name__ == "__main__":
    main()
