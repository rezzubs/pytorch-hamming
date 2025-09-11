use crate::bit_buffer::BitBuffer;

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

type Hamming64Arr = [u8; ENCODED_BYTES];

/// The encoded representation of 64 bit data.
///
/// See [`hamming_encode64`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hamming64(pub Hamming64Arr);

impl Hamming64 {
    pub const NUM_BYTES: usize = ENCODED_BYTES;
    pub const NUM_BITS: usize = Hamming64Arr::NUM_BITS;

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
    pub fn decode(mut self) -> ([u8; 8], bool) {
        let mut output_arr = [0u8; 8];

        // TODO: Keep track of failed corrections.
        let success = self.correct_error();
        let input_arr = &self.0;

        let mut output_idx = 0;
        for input_idx in 0..Hamming64::NUM_BITS {
            if is_par_i(input_idx) {
                continue;
            }

            if input_arr.is_1(input_idx) {
                output_arr.set_1(output_idx);
            }
            // All output bits are 0 by default.

            output_idx += 1;
        }

        (output_arr, success)
    }

    /// Encode 64 bits as a hamming code.
    pub fn encode(data: [u8; 8]) -> Self {
        let mut output_arr = Self([0u8; ENCODED_BYTES]);

        let mut input_idx = 0;

        for output_idx in 3..Hamming64::NUM_BITS {
            if is_par_i(output_idx) {
                continue;
            }

            if data.is_1(input_idx) {
                output_arr.0.set_1(output_idx);
            }
            // All output bits are 0 by default.

            input_idx += 1;
        }

        let bits_to_toggle =
            u8::try_from(output_arr.error_idx()).expect("ByteArray<9> index must fit inside u8");

        let bits_to_toggle = [bits_to_toggle];

        for i in 0..ERROR_CORRECTION_BIT_COUNT {
            let parity_bit = 1 << i;

            if bits_to_toggle.is_1(i) {
                output_arr.0.flip_bit(parity_bit);
            }
        }

        if !output_arr.total_parity_is_even() {
            output_arr.set_1(0);
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
    type Target = Hamming64Arr;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Hamming64 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl BitBuffer for Hamming64 {
    const NUM_BITS: usize = Hamming64Arr::NUM_BITS;

    fn set_1(&mut self, bit_idx: usize) {
        self.0.set_1(bit_idx)
    }

    fn set_0(&mut self, bit_idx: usize) {
        self.0.set_0(bit_idx)
    }

    fn is_1(&self, bit_idx: usize) -> bool {
        self.0.is_1(bit_idx)
    }

    fn flip_bit(&mut self, bit_idx: usize) {
        self.0.flip_bit(bit_idx);
    }
}
