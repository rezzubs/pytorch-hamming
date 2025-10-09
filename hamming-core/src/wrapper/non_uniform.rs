use crate::prelude::*;

/// Gives a [`BitBuffer`] implementation to sequences where the items cannot satisfy
/// [`SizedBitBuffer`].
///
/// If a sequence satisfies [`BitBuffer`] by itself then that implementation should always be
/// preferred. This wrapper should be used as a last resort due to performance reasons.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Hash)]
pub struct NonUniformSequence<I>(pub I);

impl<I, T> NonUniformSequence<I>
where
    for<'a> &'a I: IntoIterator<Item = &'a T>,
    T: BitBuffer,
{
    /// Return a pair of the index of the item + the index of the bit inside the item.
    fn inner_index(&self, index: usize) -> (usize, usize) {
        let mut start_of_current = 0;
        for (i, buffer) in self.0.into_iter().enumerate() {
            let start_of_next = start_of_current + buffer.num_bits();
            if index < start_of_next {
                return (i, index - start_of_current);
            }
            start_of_current = start_of_next
        }

        panic!("out of bounds");
    }
}

impl<I, T> BitBuffer for NonUniformSequence<I>
where
    for<'a> &'a I: IntoIterator<Item = &'a T>,
    I: std::ops::IndexMut<usize, Output = T>,
    T: BitBuffer,
{
    fn num_bits(&self) -> usize {
        self.0.into_iter().map(|x| x.num_bits()).sum()
    }

    fn set_1(&mut self, bit_index: usize) {
        let (outer, inner) = self.inner_index(bit_index);
        self.0[outer].set_1(inner);
    }

    fn set_0(&mut self, bit_index: usize) {
        let (outer, inner) = self.inner_index(bit_index);
        self.0[outer].set_0(inner);
    }

    fn is_1(&self, bit_index: usize) -> bool {
        let (outer, inner) = self.inner_index(bit_index);
        self.0[outer].is_1(inner)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        let (outer, inner) = self.inner_index(bit_index);
        self.0[outer].flip_bit(inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is() {
        let buffer = NonUniformSequence([
            Vec::from([0u8, 1u8]),
            Vec::from([0b10u8]),
            Vec::from([1u8, 0b10000000u8]),
        ]);

        for i in 0..=7 {
            assert!(buffer.is_0(i));
        }

        assert!(buffer.is_1(8));

        for i in 9..=16 {
            assert!(buffer.is_0(i));
        }

        assert!(buffer.is_1(17));

        for i in 18..=23 {
            assert!(buffer.is_0(i));
        }

        assert!(buffer.is_1(24));

        for i in 25..=38 {
            assert!(buffer.is_0(i))
        }

        assert!(buffer.is_1(39))
    }
}
