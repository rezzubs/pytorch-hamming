use crate::prelude::*;

/// Check if an index is reserved for parity (a power of two or 0).
fn is_par_i(i: usize) -> bool {
    if i == 0 {
        return true;
    }
    (i & (i - 1)) == 0
}

/// Get corresponding number of bits required for error correction for a buffer with length
/// `source_length`.
pub fn num_error_correction_bits(num_data_bits: usize) -> usize {
    usize::try_from(num_data_bits.ilog2())
        .expect("It fit into usize before and it won't get larger after ilog2")
        + 1
}

/// Get the number of total bits that are required to encode a buffer with length `source_length`.
pub fn num_encoded_bits(num_data_bits: usize) -> usize {
    // +1 for the 0th double error detection bit.
    num_data_bits + num_error_correction_bits(num_data_bits) + 1
}

/// Get the index of a flipped bit in an encoded buffer in case of a single bit flip.
///
/// 0 marks a successful case.
pub fn error_index<T>(buffer: &T) -> usize
where
    T: BitBuffer,
{
    use std::ops::BitXor;

    buffer
        .bits()
        .enumerate()
        .filter_map(|(i, bit_is_high)| bit_is_high.then_some(i))
        .fold(0, |acc, x| acc.bitxor(x))
}

/// Correct any single bit flip error in an encoded buffer.
///
/// Returns `true` if the correction was successful. Note that the function
/// will incorrectly try to correct any odd number of faults other than 1.
/// For an even number of faults, `false` is always returned.
///
/// If bit 0 is flipped we mark the result as `false` even if it's the only
/// faulty one because this case is indistinguishable from the even number
/// of faults case.
pub fn correct_error<T>(buffer: &mut T) -> bool
where
    T: BitBuffer,
{
    // The parity check provides double error detection. During encoding the
    // 0th bit is set so the parity across all bits is even. For single bit
    // errors we expect an odd parity.
    match (error_index(buffer), buffer.total_parity_is_even()) {
        // We couldn't find an error location and the total parity has not
        // changed.
        (0, true) => true,
        // We're not detecting any errors but the parity changed therefore
        // there must be multiple errors which are canceling each other out
        (0, false) => false,
        // We found an error location but the parity didn't change therefore
        // there must be two or more errors.
        (_, true) => false,
        // If only one of our protected bits flipped it will cause the error
        // index to be in our protected range.
        (e, false) if e >= buffer.num_bits() => false,
        // We found an error location and the parity changed which means we
        // either have 1 error which we will attempt to correct or an
        // undetectable odd number of errors.
        (e, false) => {
            buffer.flip_bit(e);
            true
        }
    }
}

/// Encode the `source` buffer as a hamming code inside the `dest` buffer.
///
/// # Panics
///
/// If the number of bits in `dest` doesn't match encoded `source`. Use [`num_encoded_bits`] to
/// verify the length.
pub fn encode_into<S, D>(source: &S, dest: &mut D)
where
    S: BitBuffer,
    D: BitBuffer,
{
    let num_error_correction_bits = num_error_correction_bits(source.num_bits());
    let num_encoded_bits = num_encoded_bits(source.num_bits());
    assert_eq!(dest.num_bits(), num_encoded_bits);

    let mut input_index = 0;
    // NOTE: starting from 3 because 0, 1, 2 are all reserved for parity.
    for output_index in 3..num_encoded_bits {
        if is_par_i(output_index) {
            continue;
        }

        if source.is_1(input_index) {
            dest.set_1(output_index);
        } else {
            dest.set_0(output_index);
        }

        input_index += 1;
    }

    let bits_to_toggle = u64::try_from(error_index(dest)).expect("error index out of bounds");

    for i in 0..num_error_correction_bits {
        let parity_bit = 1 << i;

        if bits_to_toggle.is_1(i) {
            dest.flip_bit(parity_bit);
        }
    }

    if !dest.total_parity_is_even() {
        dest.set_1(0);
    }
}

/// Decode an encoded buffer to the original representation.
///
/// The decoding process tries to correct single-bit-errors which modifies the `source` buffer.
///
/// # Panics
///
/// If `dest` doesn't have enough space for the decoded `source`.
pub fn decode_into<S, D>(source: &mut S, dest: &mut D) -> bool
where
    S: BitBuffer,
    D: BitBuffer,
{
    let success = correct_error(source);

    let mut output_index = 0;
    for input_index in 0..source.num_bits() {
        if is_par_i(input_index) {
            continue;
        }

        if source.is_1(input_index) {
            dest.set_1(output_index);
        } else {
            dest.set_0(output_index);
        }

        output_index += 1;
    }

    success
}
