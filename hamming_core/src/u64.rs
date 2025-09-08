use crate::byte_array::ByteArray;

/// Bit's reserved for error correction. All powers of two.
const ERROR_CORRECTION_BIT_COUNT: usize = 7;

/// Number of bytes used for the encoded format.
const ENCODED_BYTES: usize = 9;

/// Check if an index is reserved for parity (a power of two).
fn is_par_i(i: usize) -> bool {
    if i == 0 {
        return true;
    }
    (i & (i - 1)) == 0
}

/// The encoded representation of 64 bit data.
///
/// See [`hamming_encode64`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hamming64(pub ByteArray<ENCODED_BYTES>);

impl Hamming64 {
    pub const NUM_BYTES: usize = ENCODED_BYTES;
    pub const NUM_BITS: usize = ByteArray::<ENCODED_BYTES>::NUM_BITS;

    /// Correct any single bit flip error in self.
    ///
    /// Returns `true` if the correction was successful. Note that the function
    /// will incorrectly try to correct any odd number of faults other than 1.
    /// For an even number of faults, `false` is always returned.
    ///
    /// If bit 0 is flipped we mark the result as `false` even if it's the only
    /// faulty one because this case is indistinguishable from the even number
    /// of faults case.
    pub fn correct_error(&mut self) -> bool {
        // The parity check provides double error detection. During encoding the
        // 0th bit is set so the parity across all bits is even. For single bit
        // errors we expect an odd parity.
        match (self.error_idx(), self.0.total_parity_is_even()) {
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
            (e, false) if e >= Self::NUM_BITS => false,
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
    /// The second returned value will be false for failed error correction. See
    /// [`Self::correct_error`] for details.
    pub fn decode(mut self) -> (ByteArray<8>, bool) {
        let mut output_arr: ByteArray<8> = ByteArray::new();

        // TODO: Keep track of failed corrections.
        let success = self.correct_error();
        let input_arr = &self.0;

        let mut output_idx = 0;
        for input_idx in 0..Hamming64::NUM_BITS {
            if is_par_i(input_idx) {
                continue;
            }

            if input_arr.bit_is_high(input_idx) {
                output_arr.set_bit_high(output_idx);
            }
            // All output bits are 0 by default.

            output_idx += 1;
        }

        (output_arr, success)
    }

    /// Encode 64 bits as a hamming code.
    pub fn encode(data: impl Into<ByteArray<8>>) -> Self {
        let input_arr: ByteArray<8> = data.into();
        let mut output_arr = Self(ByteArray::new());

        let mut input_idx = 0;

        for output_idx in 3..Hamming64::NUM_BITS {
            if is_par_i(output_idx) {
                continue;
            }

            if input_arr.bit_is_high(input_idx) {
                output_arr.0.set_bit_high(output_idx);
            }
            // All output bits are 0 by default.

            input_idx += 1;
        }

        let bits_to_toggle =
            u8::try_from(output_arr.error_idx()).expect("ByteArray<9> index must fit inside u8");

        let bits_to_toggle = ByteArray::from(bits_to_toggle);

        for i in 0..ERROR_CORRECTION_BIT_COUNT {
            let parity_bit = 1 << i;

            if bits_to_toggle.bit_is_high(i) {
                output_arr.0.flip_bit(parity_bit);
            }
        }

        if !output_arr.total_parity_is_even() {
            output_arr.set_bit_high(0);
        }

        output_arr
    }

    /// Get the index of a flipped bit in case of a single bit flip.
    ///
    /// 0 marks a successful case.
    pub fn error_idx(&self) -> usize {
        use std::ops::BitXor;

        self.0
            .bits()
            .enumerate()
            .filter_map(|(i, bit_is_high)| bit_is_high.then_some(i))
            .fold(0, |acc, x| acc.bitxor(x))
    }
}

impl std::ops::Deref for Hamming64 {
    type Target = ByteArray<ENCODED_BYTES>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Hamming64 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
