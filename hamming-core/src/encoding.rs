use crate::BitBuffer;

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

pub trait Encodable<D, O>: BitBuffer
where
    D: Decodable<O> + Init,
    O: BitBuffer + Init,
{
    /// Encode the buffer with a hamming code.
    fn encode(&self) -> D {
        let mut output_buffer = D::init();

        let mut input_idx = 0;

        // NOTE: starting from 3 because 0, 1, 2 are all reserved for parity.
        for output_idx in 3..D::NUM_ENCODED_BITS {
            if is_par_i(output_idx) {
                continue;
            }

            if self.is_1(input_idx) {
                output_buffer.set_1(output_idx);
            } else {
                output_buffer.set_0(output_idx);
            }

            input_idx += 1;
        }

        let bits_to_toggle =
            u8::try_from(output_buffer.error_idx()).expect("ByteArray<9> index must fit inside u8");

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

pub trait Decodable<O>: BitBuffer
where
    O: BitBuffer + Init,
{
    const NUM_ENCODED_BITS: usize;
    const NUM_ERROR_CORRECTION_BITS: usize;

    /// Get the index of a flipped bit in case of a single bit flip.
    ///
    /// 0 marks a successful case.
    fn error_idx(&self) -> usize {
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
        match (self.error_idx(), self.total_parity_is_even()) {
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

        let mut output_idx = 0;
        for input_idx in 0..Self::NUM_ENCODED_BITS {
            if is_par_i(input_idx) {
                continue;
            }

            if self.is_1(input_idx) {
                output_buffer.set_1(output_idx);
            } else {
                output_buffer.set_0(output_idx);
            }

            output_idx += 1;
        }

        (output_buffer, success)
    }
}

impl Init for [u8; 8] {
    fn init() -> Self {
        Default::default()
    }
}

impl Init for [u8; 9] {
    fn init() -> Self {
        Default::default()
    }
}

impl Encodable<[u8; 9], [u8; 8]> for [u8; 8] {}

impl Decodable<[u8; 8]> for [u8; 9] {
    // 64 data + 7 SEC + 1 DED.
    const NUM_ENCODED_BITS: usize = 64 + 7 + 1;

    const NUM_ERROR_CORRECTION_BITS: usize = 7;
}

impl Init for [u8; 16] {
    fn init() -> Self {
        Default::default()
    }
}

impl Init for [u8; 18] {
    fn init() -> Self {
        Default::default()
    }
}

impl Encodable<[u8; 18], [u8; 16]> for [u8; 16] {}

impl Decodable<[u8; 16]> for [u8; 18] {
    // 128 data + 8 SEC + 1 DED.
    const NUM_ENCODED_BITS: usize = 128 + 8 + 1;

    const NUM_ERROR_CORRECTION_BITS: usize = 8;
}

impl Init for [u8; 32] {
    fn init() -> Self {
        Default::default()
    }
}

impl Init for [u8; 34] {
    fn init() -> Self {
        [0u8; 34]
    }
}

impl Encodable<[u8; 34], [u8; 32]> for [u8; 32] {}

impl Decodable<[u8; 32]> for [u8; 34] {
    // 128 data + 8 SEC + 1 DED.
    const NUM_ENCODED_BITS: usize = 128 + 8 + 1;

    const NUM_ERROR_CORRECTION_BITS: usize = 8;
}
