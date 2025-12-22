//! A single error correction, double error detection encoding scheme for bit buffers.

use crate::prelude::*;

/// Check if an index is reserved for parity (a power of two or 0).
#[must_use]
pub fn is_par_i(i: usize) -> bool {
    if i == 0 {
        return true;
    }
    (i & (i - 1)) == 0
}

/// Get corresponding number of bits required for error correction for a buffer with length
/// `source_length`.
///
/// Returns None if `data_bits_count` is 0.
#[must_use]
pub fn error_correction_bits_count(data_bits_count: usize) -> Option<usize> {
    if data_bits_count == 0 {
        return None;
    }

    // NOTE: 2 is the minimum possible number of parity bits.
    let mut parity_bits = 2u32;
    loop {
        let max_data_bits_per_parity_bits = (2u32.pow(parity_bits) - parity_bits - 1) as usize;
        if data_bits_count <= max_data_bits_per_parity_bits {
            return Some(parity_bits as usize);
        }
        parity_bits += 1
    }
}

/// Get the number of total bits that are required to encode a buffer with length `source_length`.
///
/// Returns None if `data_bits_count` is 0.
#[must_use]
pub fn encoded_bits_count(data_bits_count: usize) -> Option<usize> {
    // +1 for the 0th double error detection bit.
    Some(data_bits_count + error_correction_bits_count(data_bits_count)? + 1)
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
        (e, false) if e >= buffer.bits_count() => false,
        // We found an error location and the parity changed which means we
        // either have 1 error which we will attempt to correct or an
        // undetectable odd number of errors.
        (e, false) => {
            buffer.flip_bit(e);
            true
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EncodeError {
    #[error("The source cannot be empty")]
    SourceEmpty,
    #[error("The destination buffer should have {expected} bits based on the source, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("Expected the destination buffer to have {expected} bytes, got {actual}")]
pub struct LengthMismatch {
    pub expected: usize,
    pub actual: usize,
}

/// Encode the `source` buffer as a hamming code inside the `dest` buffer.
pub fn encode_into<S, D>(source: &S, dest: &mut D) -> Result<(), EncodeError>
where
    S: BitBuffer,
    D: BitBuffer,
{
    let error_correction_bits_count =
        error_correction_bits_count(source.bits_count()).ok_or(EncodeError::SourceEmpty)?;
    let encoded_bits_count = encoded_bits_count(source.bits_count()).expect("already checked");

    if dest.bits_count() != encoded_bits_count {
        return Err(EncodeError::LengthMismatch {
            expected: encoded_bits_count,
            actual: dest.bits_count(),
        });
    }

    let mut input_index = 0;
    // NOTE: starting from 3 because 0, 1, 2 are all reserved for parity.
    for output_index in 3..encoded_bits_count {
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

    for i in 0..error_correction_bits_count {
        let parity_bit = 1 << i;

        if bits_to_toggle.is_1(i) {
            dest.flip_bit(parity_bit);
        }
    }

    if !dest.total_parity_is_even() {
        dest.set_1(0);
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecodeError {
    #[error("The destination buffer cannot be empty")]
    DestEmpty,
    #[error(
        "The encoded buffer should have {expected} bits based on the destination, got {actual}"
    )]
    LengthMismatch { expected: usize, actual: usize },
}

/// Decode an encoded buffer to the original representation.
///
/// The decoding process tries to correct single-bit-errors which modifies the `source` buffer.
///
/// Returns `false` if two-or-more-bit errors were detected.
pub fn decode_into<S, D>(source: &mut S, dest: &mut D) -> Result<bool, DecodeError>
where
    S: BitBuffer + std::fmt::Debug,
    D: BitBuffer,
{
    let success = correct_error(source);

    let encoded_bits_count = encoded_bits_count(dest.bits_count()).ok_or(DecodeError::DestEmpty)?;

    if source.bits_count() != encoded_bits_count {
        return Err(DecodeError::LengthMismatch {
            expected: encoded_bits_count,
            actual: source.bits_count(),
        });
    }

    let mut output_index = 0;
    for input_index in 3..encoded_bits_count {
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

    Ok(success)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_i() {
        assert!(is_par_i(0));
        assert!(is_par_i(1));
        assert!(is_par_i(2));
        assert!(!is_par_i(3));
        assert!(is_par_i(4));
        assert!(!is_par_i(5));
        assert!(!is_par_i(6));
        assert!(!is_par_i(7));
        assert!(is_par_i(8));
        assert!(!is_par_i(9));
        assert!(!is_par_i(10));
        assert!(!is_par_i(11));
        assert!(!is_par_i(12));
        assert!(!is_par_i(13));
        assert!(!is_par_i(14));
        assert!(!is_par_i(15));
        assert!(is_par_i(16));
    }

    #[test]
    fn bits_count() {
        assert_eq!(error_correction_bits_count(1).unwrap(), 2);
        for i in 3..=4 {
            assert_eq!(error_correction_bits_count(i).unwrap(), 3);
        }
        for i in 5..=11 {
            assert_eq!(error_correction_bits_count(i).unwrap(), 4);
        }
        for i in 12..=26 {
            assert_eq!(error_correction_bits_count(i).unwrap(), 5);
        }
        for i in 27..=57 {
            assert_eq!(error_correction_bits_count(i).unwrap(), 6);
        }
        for i in 58..=120 {
            assert_eq!(error_correction_bits_count(i).unwrap(), 7);
        }
        for i in 121..=247 {
            assert_eq!(error_correction_bits_count(i).unwrap(), 8);
        }
        for i in 248..=502 {
            assert_eq!(error_correction_bits_count(i).unwrap(), 9);
        }
        assert_eq!(error_correction_bits_count(512).unwrap(), 10);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use std::ops::RangeInclusive;

    const RANGE: RangeInclusive<usize> = 1..=512;

    proptest! {
        #[test]
        fn encode_decode_u8_zero_fault(
            (buf, mut decoded) in (RANGE).prop_flat_map(|len| {
                (
                    prop::collection::vec(any::<u8>(), len),
                    prop::collection::vec(any::<u8>(), len),
                )
            })
        ) {
            let mut encoded = buf.encode().unwrap();

            let success = decode_into(&mut encoded, &mut decoded).unwrap();
            assert!(success);

            assert_eq!(buf, decoded);
        }

        #[test]
        fn encode_decode_u8_single_fault(
            ((buf, fault), mut decoded) in (RANGE).prop_flat_map(|len| {
                (
                    prop::collection::vec(any::<u8>(), len).prop_flat_map(|v| {
                        let fault_max = 8 * v.len();
                        (Just(v), 1..=fault_max)
                    }),
                    prop::collection::vec(any::<u8>(), len),
                )
            })
        ) {
            let mut encoded = match buf.encode() {
                Ok(encoded) => encoded,
                Err(crate::bit_buffer::EncodeError::Empty) => return Ok(()),
            };

            encoded.flip_bit(fault);

            let success = decode_into(&mut encoded, &mut decoded).unwrap();
            assert!(success);

            assert_eq!(buf, decoded);
        }

        #[test]
        fn encode_decode_f32_zero_fault(
            (buf, mut decoded) in (RANGE).prop_flat_map(|len| {
                (
                    prop::collection::vec(any::<f32>(), len),
                    prop::collection::vec(any::<f32>(), len),
                )
            })
        ) {
            let mut encoded = buf.encode().unwrap();

            let success = decode_into(&mut encoded, &mut decoded).unwrap();
            assert!(success);

            assert_eq!(buf, decoded);
        }

        #[test]
        fn encode_decode_f32_single_fault(
            ((buf, fault), mut decoded) in (RANGE).prop_flat_map(|len| {
                (
                    prop::collection::vec(any::<f32>(), len).prop_flat_map(|v| {
                        let fault_max = 8 * v.len();
                        (Just(v), 1..=fault_max)
                    }),
                    prop::collection::vec(any::<f32>(), len),
                )
            })
        ) {
            let mut encoded = buf.encode().unwrap();

            encoded.flip_bit(fault);

            let success = decode_into(&mut encoded, &mut decoded).unwrap();
            assert!(success);

            assert_eq!(buf, decoded);
        }

        #[test]
        fn encode_decode_f32_two_faults(
            ((buf, (fault1, fault2)), mut decoded) in (RANGE).prop_flat_map(|len| (
                prop::collection::vec(any::<f32>(), len).prop_flat_map(|v| {
                    let fault_max = 8 * v.len();
                    (Just(v), (1..=fault_max, 1..=fault_max).prop_filter("must differ", |(a, b)| a != b))
                }),
                prop::collection::vec(any::<f32>(), len),
            ))
        ) {
            let encoded = buf.encode().unwrap();

            let mut faulty = encoded.clone();
            faulty.flip_bit(fault1);
            faulty.flip_bit(fault2);

            assert_ne!(faulty, encoded);

            // A two bit flip will always be marked as unsuccessful even if in
            // reality the data matches. See [`correct_error`].
            let success = decode_into(&mut faulty, &mut decoded).unwrap();
            assert!(!success);

            // If only parity bits were hit then the original data is safe. The decoding
            // algorithm must still call a failure because there's no way to tell which bits
            // were hit.
            if is_par_i(fault1) && is_par_i(fault2) {
                assert_eq!(buf, decoded);
            }        }
    }
}
