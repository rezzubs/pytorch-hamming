//! [`BitBuffer`] implementations for unsigned integers.

use super::{BitBuffer, SizedBitBuffer};

macro_rules! float_impl {
    ($t:ty, $bits:expr) => {
        impl SizedBitBuffer for $t {
            const NUM_BITS: usize = $bits;
        }

        impl BitBuffer for $t {
            fn num_bits(&self) -> usize {
                Self::NUM_BITS
            }

            fn set_1(&mut self, bit_index: usize) {
                let mut unsigned = self.to_bits();
                unsigned.set_1(bit_index);
                *self = <$t>::from_bits(unsigned)
            }

            fn set_0(&mut self, bit_index: usize) {
                let mut unsigned = self.to_bits();
                unsigned.set_0(bit_index);
                *self = <$t>::from_bits(unsigned)
            }

            fn is_1(&self, bit_index: usize) -> bool {
                let unsigned = self.to_bits();
                unsigned.is_1(bit_index)
            }

            fn flip_bit(&mut self, bit_index: usize) {
                let mut unsigned = self.to_bits();
                unsigned.flip_bit(bit_index);
                *self = <$t>::from_bits(unsigned)
            }
        }
    };
}

float_impl!(f32, 32);
float_impl!(f64, 64);
