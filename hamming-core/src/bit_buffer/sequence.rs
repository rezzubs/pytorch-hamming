//! [`BitBuffer`] implementations for sequences.

use super::{BitBuffer, SizedBitBuffer};

impl<T> BitBuffer for [T]
where
    T: SizedBitBuffer,
{
    /// Total number of bits in the buffer.
    fn num_bits(&self) -> usize {
        self.len() * T::NUM_BITS
    }

    fn set_1(&mut self, bit_index: usize) {
        assert!(bit_index < self.num_bits());
        let item_index = bit_index / T::NUM_BITS;
        self[item_index].set_1(bit_index % T::NUM_BITS);
    }

    fn set_0(&mut self, bit_index: usize) {
        assert!(bit_index < self.num_bits());
        let item_index = bit_index / T::NUM_BITS;
        self[item_index].set_0(bit_index % T::NUM_BITS);
    }

    fn is_1(&self, bit_index: usize) -> bool {
        assert!(bit_index < self.num_bits());
        let item_index = bit_index / T::NUM_BITS;
        self[item_index].is_1(bit_index % T::NUM_BITS)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        assert!(bit_index < self.num_bits());
        let item_index = bit_index / T::NUM_BITS;
        self[item_index].flip_bit(bit_index % T::NUM_BITS);
    }
}

impl<const N: usize, T> SizedBitBuffer for [T; N]
where
    T: SizedBitBuffer,
{
    const NUM_BITS: usize = N * T::NUM_BITS;
}

impl<const N: usize, T> BitBuffer for [T; N]
where
    T: SizedBitBuffer,
{
    fn num_bits(&self) -> usize {
        self.as_slice().num_bits()
    }

    fn set_1(&mut self, bit_index: usize) {
        self.as_mut_slice().set_1(bit_index);
    }

    fn set_0(&mut self, bit_index: usize) {
        self.as_mut_slice().set_0(bit_index);
    }

    fn is_1(&self, bit_index: usize) -> bool {
        self.as_slice().is_1(bit_index)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        self.as_mut_slice().flip_bit(bit_index);
    }
}
