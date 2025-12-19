use crate::bit_buffer::OutOfBounds;
use crate::prelude::*;

/// Gives a [`BitBuffer`] implementation to sequences where the items cannot satisfy
/// [`SizedBitBuffer`].
///
/// If the number of bits is only runtime known but still expected to be the same for all elements
/// then [`crate::wrapper::UniformSequence`] should be used for better performance.
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
    fn inner_bit_index(&self, index: usize) -> Result<(usize, usize), OutOfBounds> {
        let mut start_of_current = 0;
        for (i, buffer) in self.0.into_iter().enumerate() {
            let start_of_next = start_of_current + buffer.bits_count();
            if index < start_of_next {
                return Ok((i, index - start_of_current));
            }
            start_of_current = start_of_next
        }

        Err(OutOfBounds)
    }
}

impl<I, T> NonUniformSequence<I>
where
    for<'a> &'a I: IntoIterator<Item = &'a T>,
    T: ByteChunkedBitBuffer,
{
    /// Return a pair of the index of the item + the index of the byte inside the item.
    fn inner_byte_index(&self, index: usize) -> Result<(usize, usize), OutOfBounds> {
        let mut start_of_current = 0;
        for (i, buffer) in self.0.into_iter().enumerate() {
            let start_of_next = start_of_current + buffer.bytes_count();
            if index < start_of_next {
                return Ok((i, index - start_of_current));
            }
            start_of_current = start_of_next
        }

        Err(OutOfBounds)
    }
}

impl<I, T> BitBuffer for NonUniformSequence<I>
where
    for<'a> &'a I: IntoIterator<Item = &'a T>,
    I: std::ops::IndexMut<usize, Output = T>,
    T: BitBuffer,
{
    fn bits_count(&self) -> usize {
        self.0.into_iter().map(|x| x.bits_count()).sum()
    }

    fn set_1(&mut self, bit_index: usize) {
        let (outer, inner) = self.inner_bit_index(bit_index).expect("out of bounds");
        self.0[outer].set_1(inner);
    }

    fn set_0(&mut self, bit_index: usize) {
        let (outer, inner) = self.inner_bit_index(bit_index).expect("out of bounds");
        self.0[outer].set_0(inner);
    }

    fn is_1(&self, bit_index: usize) -> bool {
        let (outer, inner) = self.inner_bit_index(bit_index).expect("out of bounds");
        self.0[outer].is_1(inner)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        let (outer, inner) = self.inner_bit_index(bit_index).expect("out of bounds");
        self.0[outer].flip_bit(inner)
    }
}

impl<I, T> ByteChunkedBitBuffer for NonUniformSequence<I>
where
    for<'a> &'a I: IntoIterator<Item = &'a T>,
    I: std::ops::IndexMut<usize, Output = T>,
    T: ByteChunkedBitBuffer,
{
    fn bytes_count(&self) -> usize {
        self.0.into_iter().map(|x| x.bytes_count()).sum()
    }

    fn get_byte(&self, n: usize) -> u8 {
        let (outer, inner) = self.inner_byte_index(n).expect("out of bounds");
        self.0[outer].get_byte(inner)
    }

    fn set_byte(&mut self, n: usize, value: u8) {
        let (outer, inner) = self.inner_byte_index(n).expect("out of bounds");
        self.0[outer].set_byte(inner, value)
    }
}

#[cfg(test)]
mod tests {
    use crate::bit_buffer::CopyIntoResult;

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

    #[test]
    fn bounds() {
        let buffer = NonUniformSequence(vec![0u16; 9]);

        assert!(buffer.inner_bit_index(buffer.bits_count()).is_err());
        assert!(buffer.inner_bit_index(buffer.bits_count() - 1).is_ok());
        assert!(buffer.inner_byte_index(buffer.bytes_count()).is_err());
        assert!(buffer.inner_byte_index(buffer.bytes_count() - 1).is_ok());
    }

    #[test]
    fn copy_into() {
        let a_true: NonUniformSequence<Vec<Vec<u8>>> = NonUniformSequence(vec![
            vec![123],
            vec![13, 255, 8],
            vec![0, 1],
            vec![255],
            vec![],
            vec![0],
        ]);
        let b_true = [
            u16::from_le_bytes([123, 13]),
            u16::from_le_bytes([255, 8]),
            u16::from_le_bytes([0, 1]),
            u16::from_le_bytes([255, 0]),
        ];

        let mut b: Vec<u16> = vec![0; 4];

        let result = a_true.copy_into(&mut b);
        assert_eq!(result, CopyIntoResult::done(a_true.bits_count()));
        assert_eq!(result.units_copied, b_true.bits_count());

        assert_eq!(b, b_true);

        let mut a: NonUniformSequence<Vec<Vec<u8>>> = NonUniformSequence(vec![
            vec![0],
            vec![0, 0, 0],
            vec![0, 0],
            vec![0],
            vec![],
            vec![0],
        ]);

        let result = b_true.copy_into(&mut a);
        assert_eq!(result, CopyIntoResult::done(a_true.bits_count()));
        assert_eq!(result.units_copied, b_true.bits_count());

        assert_eq!(a, a_true);
    }

    #[test]
    fn copy_into_chunked() {
        let a_true: NonUniformSequence<Vec<Vec<u8>>> = NonUniformSequence(vec![
            vec![123],
            vec![13, 255, 8],
            vec![0, 1],
            vec![255],
            vec![],
            vec![0],
        ]);
        let b_true = [
            u16::from_le_bytes([123, 13]),
            u16::from_le_bytes([255, 8]),
            u16::from_le_bytes([0, 1]),
            u16::from_le_bytes([255, 0]),
        ];

        let mut b: Vec<u16> = vec![0; 4];

        let result = a_true.copy_into_chunked(&mut b);
        assert_eq!(result.units_copied, a_true.bytes_count());
        assert_eq!(result.units_copied, b_true.bytes_count());

        assert_eq!(b, b_true);

        let mut a: NonUniformSequence<Vec<Vec<u8>>> = NonUniformSequence(vec![
            vec![0],
            vec![0, 0, 0],
            vec![0, 0],
            vec![0],
            vec![],
            vec![0],
        ]);

        let result = a_true.copy_into_chunked(&mut a);
        assert_eq!(result.units_copied, a_true.bytes_count());
        assert_eq!(result.units_copied, b_true.bytes_count());

        assert_eq!(a, a_true);
    }

    #[test]
    fn flip_bit() {
        let mut buf = NonUniformSequence(vec![vec![0u8, 0], vec![1], vec![0b10]]);

        assert_eq!(buf.0[0][0], 0);
        buf.flip_bit(0);
        assert_eq!(buf.0[0][0], 1);

        assert_eq!(buf.0[1][0], 1);
        buf.flip_bit(16);
        assert_eq!(buf.0[1][0], 0);

        assert_eq!(buf.0[2][0], 2);
        // Second bit of the last element
        buf.flip_bit(25);
        assert_eq!(buf.0[2][0], 0);
    }
}
