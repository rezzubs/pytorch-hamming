"""Encoding and decoding a single value."""

from pytorch_ecc import hamming_encode
from pytorch_ecc._hamming import hamming_decode

import torch


def main():
    initial = torch.tensor([0.875])
    print("initial: ", initial)
    encoded = hamming_encode(initial)
    print("encoded: ", encoded)
    # We're toggling the bit after the sign bit.
    # Without correction this would make the value into 2.9775e+38
    encoded = encoded ^ torch.tensor([1 << 37])
    print("faulty:  ", encoded)
    decoded = hamming_decode(encoded)
    print("decoded: ", decoded)
    assert torch.equal(decoded, initial)


if __name__ == "__main__":
    main()
