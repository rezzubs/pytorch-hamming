//! Traits for hamming encoding [`BitBuffer`]s.

mod impls;
#[cfg(test)]
mod tests;

use crate::{BitBuffer, SizedBitBuffer};

/// Check if an index is reserved for parity (a power of two or 0).
fn is_par_i(i: usize) -> bool {
    if i == 0 {
        return true;
    }
    (i & (i - 1)) == 0
}

/// A trait for getting an arbitrary buffer from a type.
///
/// The actual value doesn't matter. This is just used as the default value in other functions. Can
/// be delegated to [`Default::default`] if your type implements it.
pub trait Init {
    /// Return a value.
    fn init() -> Self;
}

/// Buffers which can be encoded as a hamming code.
///
/// All BitBuffers can implement this without restrictions. This trait is always paired with
/// [`Decodable`]. To be able to encode a buffer you must also define the decodable variant.
///
/// # Type parameters
///
/// - `D` is the buffer which is going to store the encoded data. Must be able to store at least as
///   many bits as the encoded data (`original.num_bits() + original.num_bits().ilog2() + 2`). There
///   is no way to verify this at compile time. If the buffer doesn't have enough space then the
///   behavior is undefined, most likely will cause a crash.
/// - `O` is the buffer which is going to store the decoded data. This can be the same type as the
///   original buffer as long as the number of bits in the buffer is known at compile time - see
///   [`SizedBitBuffer`].
///
/// Both `D` and `O` need to implement [`Init`] which will be used to create a blank slate value
/// during encoding/decoding.
///
/// [`PaddedBuffer`] Might be useful for `D` if the number of bytes in the result doesn't fill the
/// whole `D` buffer. A newtype wrapper may need to be used outside of this crate.
///
/// # Example for implementing `Encoding` and `Decoding`.
///
/// Assuming both `D` and `O` implement [`Init`].
///
/// ```ignore
/// type EncodedU8 = PaddedBuffer<[u8; 2], 13, 3>;
///
/// impl Encodable<EncodedU8, u8> for u8 {}
///
/// impl Decodable<u8> for EncodedU8 {}
/// ```
pub trait Encodable<D, O>: BitBuffer
where
    D: Decodable<O> + Init,
    O: SizedBitBuffer + Init,
{
    /// Encode the buffer with a hamming code.
    fn encode(&self) -> D {
        let mut output_buffer = D::init();

        let mut input_index = 0;

        // NOTE: starting from 3 because 0, 1, 2 are all reserved for parity.
        for output_index in 3..D::NUM_ENCODED_BITS {
            if is_par_i(output_index) {
                continue;
            }

            if self.is_1(input_index) {
                output_buffer.set_1(output_index);
            } else {
                output_buffer.set_0(output_index);
            }

            input_index += 1;
        }

        let bits_to_toggle = u8::try_from(output_buffer.error_index())
            .expect("ByteArray<9> index must fit inside u8");

        let bits_to_toggle = [bits_to_toggle];

        for i in 0..D::NUM_ERROR_CORRECTION_BITS {
            let parity_bit = 1 << i;

            if bits_to_toggle.is_1(i) {
                output_buffer.flip_bit(parity_bit);
            }
        }

        if !output_buffer.total_parity_is_even() {
            output_buffer.set_1(0);
        }

        output_buffer
    }
}

/// A hamming encoded buffer that can be decoded.
///
/// See [`Encodable`] for details.
pub trait Decodable<O>: SizedBitBuffer
where
    O: SizedBitBuffer + Init,
{
    const NUM_ERROR_CORRECTION_BITS: usize = O::NUM_BITS.ilog2() as usize + 1;
    const NUM_ENCODED_BITS: usize = O::NUM_BITS + Self::NUM_ERROR_CORRECTION_BITS + 1;

    /// Get the index of a flipped bit in case of a single bit flip.
    ///
    /// 0 marks a successful case.
    fn error_index(&self) -> usize {
        use std::ops::BitXor;

        self.bits()
            .enumerate()
            .filter_map(|(i, bit_is_high)| bit_is_high.then_some(i))
            .fold(0, |acc, x| acc.bitxor(x))
    }

    /// Correct any single bit flip error in self.
    ///
    /// Returns `true` if the correction was successful. Note that the function
    /// will incorrectly try to correct any odd number of faults other than 1.
    /// For an even number of faults, `false` is always returned.
    ///
    /// If bit 0 is flipped we mark the result as `false` even if it's the only
    /// faulty one because this case is indistinguishable from the even number
    /// of faults case.
    fn correct_error(&mut self) -> bool {
        // The parity check provides double error detection. During encoding the
        // 0th bit is set so the parity across all bits is even. For single bit
        // errors we expect an odd parity.
        match (self.error_index(), self.total_parity_is_even()) {
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
            (e, false) if e >= self.num_bits() => false,
            // We found an error location and the parity changed which means we
            // either have 1 error which we will attempt to correct or an
            // undetectable odd number of errors.
            (e, false) => {
                self.flip_bit(e);
                true
            }
        }
    }

    /// Decode into the original bit representation.
    ///
    /// [`Decodable::correct_error`] will be called on `self` so this function should only be used
    /// once.
    ///
    /// The second returned value will be false for failed error correction. See
    /// [`Decodable::correct_error`] for details.
    fn decode(&mut self) -> (O, bool) {
        let mut output_buffer = O::init();

        let success = self.correct_error();

        let mut output_index = 0;
        for input_index in 0..Self::NUM_ENCODED_BITS {
            if is_par_i(input_index) {
                continue;
            }

            if self.is_1(input_index) {
                output_buffer.set_1(output_index);
            } else {
                output_buffer.set_0(output_index);
            }

            output_index += 1;
        }

        (output_buffer, success)
    }
}
