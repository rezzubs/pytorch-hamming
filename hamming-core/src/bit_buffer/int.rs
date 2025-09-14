//! [`BitBuffer`] implementations for integers.

use super::{BitBuffer, SizedBitBuffer};

macro_rules! int_impl {
    ($t:ty, $bits:expr) => {
        impl SizedBitBuffer for $t {
            const NUM_BITS: usize = $bits;
        }

        impl BitBuffer for $t {
            fn num_bits(&self) -> usize {
                Self::NUM_BITS
            }

            fn set_1(&mut self, bit_index: usize) {
                assert!(bit_index < Self::NUM_BITS);
                *self |= 1 << bit_index
            }

            fn set_0(&mut self, bit_index: usize) {
                assert!(bit_index < Self::NUM_BITS);
                *self &= !(1 << bit_index)
            }

            fn is_1(&self, bit_index: usize) -> bool {
                assert!(bit_index < Self::NUM_BITS);
                (self & (1 << bit_index)) > 0
            }

            fn flip_bit(&mut self, bit_index: usize) {
                assert!(bit_index < Self::NUM_BITS);
                *self ^= 1 << bit_index
            }
        }
    };
}

int_impl!(u8, 8);
int_impl!(u16, 16);
int_impl!(u32, 32);
int_impl!(u64, 64);
int_impl!(u128, 128);
int_impl!(i8, 8);
int_impl!(i16, 16);
int_impl!(i32, 32);
int_impl!(i64, 64);
int_impl!(i128, 128);
