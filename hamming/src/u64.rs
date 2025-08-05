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
    pub fn correct_error(&mut self) {
        let e = self.error_idx();
        self.flip_bit(e);
    }

    /// Decode into the original bit representation.
    pub fn decode(mut self) -> ByteArray<8> {
        let mut output_arr: ByteArray<8> = ByteArray::new();

        self.correct_error();
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

        output_arr
    }

    /// Encode 64 bits as a hamming code.
    pub fn encode(data: impl Into<ByteArray<8>>) -> Self {
        let input_arr: ByteArray<8> = data.into();
        let mut output_arr = Self(ByteArray::new());

        let mut input_idx = 0;

        for output_idx in 0..Hamming64::NUM_BITS {
            if is_par_i(output_idx) {
                continue;
            }

            if input_arr.bit_is_high(input_idx) {
                output_arr.0.set_bit_high(output_idx);
            } else {
                output_arr.0.set_bit_low(output_idx);
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
