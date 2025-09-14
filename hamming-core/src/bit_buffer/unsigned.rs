//! [`BitBuffer`] implementations for unsigned integers.

use super::{BitBuffer, SizedBitBuffer};

impl SizedBitBuffer for u8 {
    const NUM_BITS: usize = 8;
}

impl BitBuffer for u8 {
    fn num_bits(&self) -> usize {
        Self::NUM_BITS
    }

    fn set_1(&mut self, bit_index: usize) {
        assert!(bit_index <= 7);
        *self |= 1 << bit_index
    }

    fn set_0(&mut self, bit_index: usize) {
        assert!(bit_index <= 7);
        *self &= !(1 << bit_index)
    }

    fn is_1(&self, bit_index: usize) -> bool {
        assert!(bit_index <= 7);
        (self & (1 << bit_index)) > 0
    }

    fn flip_bit(&mut self, bit_index: usize) {
        assert!(bit_index <= 7);
        *self ^= 1 << bit_index
    }
}
