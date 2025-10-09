use crate::prelude::*;

/// A newtype wrapper for providing a [`std::default::Default`] implementation for arrays which
/// don't have one.
// TODO: Remove this after https://github.com/rust-lang/rust/issues/61415 is stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ZeroableArray<T, const N: usize>(pub [T; N]);

impl<T, const N: usize> Default for ZeroableArray<T, N>
where
    T: Default + Copy,
{
    fn default() -> Self {
        Self([T::default(); N])
    }
}

impl<T, const N: usize> BitBuffer for ZeroableArray<T, N>
where
    T: SizedBitBuffer,
{
    fn num_bits(&self) -> usize {
        self.0.num_bits()
    }

    fn set_1(&mut self, bit_index: usize) {
        self.0.set_1(bit_index)
    }

    fn set_0(&mut self, bit_index: usize) {
        self.0.set_0(bit_index);
    }

    fn is_1(&self, bit_index: usize) -> bool {
        self.0.is_1(bit_index)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        self.0.flip_bit(bit_index);
    }
}

impl<T, const N: usize> SizedBitBuffer for ZeroableArray<T, N>
where
    T: SizedBitBuffer,
{
    const NUM_BITS: usize = <[T; N]>::NUM_BITS;
}
