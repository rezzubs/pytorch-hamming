"""Encoding/decoding for hamming codes"""

import torch

from pytorch_ecc._utils import get_bit, set_bit_high, toggle_bit


ENCODED_DATA_BITS_COUNT = 32
ENCODED_PARITY_BITS_COUNT = 6
ENCODED_BITS_COUNT = ENCODED_DATA_BITS_COUNT + ENCODED_PARITY_BITS_COUNT + 1

PARITY_BIT_INDICES = [2**i for i in range(ENCODED_PARITY_BITS_COUNT)]


def is_parity_bit(bit_index: int) -> bool:
    """Confirms whether or not a bit is a parity bit.

    Parity bits are all powers of two.
    """
    return (bit_index & (bit_index - 1)) == 0


def hamming_error_bit_index(encoded: torch.Tensor) -> torch.Tensor:
    """Return the index of the faulty bit for a hamming encoded tensor.

    0 marks a successful case. `encoded` is expected to be a tensor of
    `torch.int64` values.

    See also `hamming_encode`.

    # Example for a single value

    If the data is 0b01110 then the indices of the active (1) bits are as
    follows (right to left):

    - 0b01 (1)
    - 0b10 (2)
    - 0b11 (3)

    If we count the parity of every bit position in the previous list and
    assign 1 for odd and 0 for even we get `0b00`. XOR can be used to count
    parity in this manner. A result with a completely even parity like this is
    a correctly formatted hamming code.

    Let's now flip a single bit in the original data, I will choose the one
    with index 3. 0b01110 becomes 0b00110. Now we can follow the same process
    as before:

    - 0b01 (1)
    - 0b10 (1)

    resulting in `0b11` (3).

    This resulting value is also the index of the bit that we flipped. This
    works for any single bit flip.
    """
    if encoded.dtype != torch.int64:
        raise ValueError(f"Expected dtype torch.int64, got {encoded.dtype}")

    output = torch.zeros_like(encoded)
    for i in range(ENCODED_BITS_COUNT):
        bit_is_high = get_bit(encoded, i).to(torch.bool)
        output[bit_is_high] = output.bitwise_xor(i)[bit_is_high]
    return output


def hamming_encode(t: torch.Tensor) -> torch.Tensor:
    """Encode a tensor of float32 values into a hamming code represented as a tensor of int64 values.

    See `hamming_error_bit_index` for an in-depth example.
    Use `hamming_decode` for decoding.

    The output dtype is int64 because PyTorch doesn't have a built in datatype
    for uint64. The sign bit doesn't affect us because we only use the lower 39
    bits (32 data + 6 parity + 0th bit for double error detection)
    """

    if t.dtype != torch.float32:
        raise ValueError(f"Expected dtype torch.float32, got {t.dtype}")

    # NOTE: `view` is used to create a bitwise transmute of the floating point
    # data. Converting it to an integer format allows us to manipulate the bits
    # directly.
    #
    # The conversion to int64 is otherwise redundant but PyTorch doesn't
    # implement shifting operations for uint32 and we cannot sacrifice one bit
    # of data for the sign thus we expand to 64 bits.
    input_bits = t.view(torch.uint32).to(torch.int64)
    output_bits = torch.zeros_like(input_bits, dtype=torch.int64)

    input_i = 0
    for output_i in range(ENCODED_BITS_COUNT):
        if is_parity_bit(output_i):
            continue

        input_is_high = get_bit(input_bits, input_i).bool()
        output_bits = torch.where(
            input_is_high, set_bit_high(output_bits, output_i), output_bits
        )

        input_i += 1

    # Normally this corresponds to the index of the faulty element. Because
    # there's no fault right now we can mask the fault by toggling the parity
    # bits that correspond to the 1 bit positions for every element. Toggling
    # these bits sets up the data for actual error detection.
    #
    # Because every parity bit is a power of two, flipping a parity bit will
    # result in a flip in the same bit in the return value of `error_idx`.
    bits_to_toggle = hamming_error_bit_index(output_bits)

    for i in range(ENCODED_PARITY_BITS_COUNT):
        elements_to_toggle = get_bit(bits_to_toggle, i).to(torch.bool)
        output_bits = torch.where(
            elements_to_toggle,
            toggle_bit(output_bits, PARITY_BIT_INDICES[i]),
            output_bits,
        )

    return output_bits


def hamming_decode(t: torch.Tensor) -> torch.Tensor:
    """Decode a tensor produced by `hamming_encode` into the original float32 values.

    This function corrects any single bit error. Bits with an index higher than
    38 are ignored because those don't include the original data.
    """

    if t.dtype != torch.int64:
        raise ValueError(f"Expected dtyep torch.int64, got {t.dtype}")

    error_indices = hamming_error_bit_index(t)
    # A mask like 0b0010000 that will be used to toggle the error bit single bit.
    toggle_mask = torch.tensor(1, dtype=error_indices.dtype) << error_indices
    faulty_bits_mask = error_indices != 0

    t = torch.where(faulty_bits_mask, t ^ toggle_mask, t)

    output_bits = torch.zeros_like(t, dtype=torch.int64)

    output_i = 0
    for input_i in range(ENCODED_BITS_COUNT):
        if is_parity_bit(input_i):
            continue

        input_is_high = get_bit(t, input_i).bool()
        output_bits = torch.where(
            input_is_high, set_bit_high(output_bits, output_i), output_bits
        )
        output_i += 1

    return output_bits.to(torch.uint32).view(torch.float32)
