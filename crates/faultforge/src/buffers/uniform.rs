use crate::prelude::*;

/// Gives a [`BitBuffer`] implementation to sequences where the items cannot satisfy
/// [`SizedBitBuffer`].
///
/// All the elements of this sequence are expected to have the same number of bits. If this
/// condition cannot be upheld then [`crate::wrapper::NonUniformSequence`] should be used instead.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Hash)]
pub struct UniformSequence<T> {
    item_bits_count: usize,
    items_count: usize,
    sequence: T,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, thiserror::Error)]
#[error("Index {0} has a different length than the rest of the sequence")]
pub struct NonMatchingIndex(usize);

impl<T> UniformSequence<T> {
    pub fn new_unchecked(sequence: T, item_bits_count: usize, items_count: usize) -> Self {
        Self {
            item_bits_count,
            items_count,
            sequence,
        }
    }

    pub fn new<U>(sequence: T) -> Result<Self, NonMatchingIndex>
    where
        for<'a> &'a T: IntoIterator<Item = &'a U>,
        U: BitBuffer,
    {
        let mut item_bits_count: Option<usize> = None;
        let mut items_count = 0;
        for (i, item) in (&sequence).into_iter().enumerate() {
            match item_bits_count {
                Some(prev) => {
                    if item.bits_count() != prev {
                        return Err(NonMatchingIndex(i));
                    }
                }
                None => item_bits_count = Some(item.bits_count()),
            }
            items_count += 1;
        }

        let item_bits_count = item_bits_count.unwrap_or(0);
        if item_bits_count == 0 {
            debug_assert_eq!(
                items_count, 0,
                "Num bits is 0 but num items is {items_count}"
            );
        }

        Ok(Self::new_unchecked(sequence, item_bits_count, items_count))
    }

    pub fn item_bits_count(&self) -> usize {
        self.item_bits_count
    }

    pub fn items_count(&self) -> usize {
        self.items_count
    }

    pub fn inner(&self) -> &T {
        &self.sequence
    }

    #[cfg(test)]
    pub(crate) fn inner_mut(&mut self) -> &mut T {
        &mut self.sequence
    }

    pub fn into_inner(self) -> T {
        self.sequence
    }
}

impl<T, U> BitBuffer for UniformSequence<T>
where
    T: std::ops::IndexMut<usize, Output = U>,
    U: BitBuffer,
{
    fn bits_count(&self) -> usize {
        self.items_count * self.item_bits_count
    }

    fn set_1(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.bits_count());
        let item_index = bit_index / self.item_bits_count;
        self.sequence[item_index].set_1(bit_index % self.item_bits_count)
    }

    fn set_0(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.bits_count());
        let item_index = bit_index / self.item_bits_count;
        self.sequence[item_index].set_0(bit_index % self.item_bits_count)
    }

    fn is_1(&self, bit_index: usize) -> bool {
        debug_assert!(bit_index < self.bits_count());
        let item_index = bit_index / self.item_bits_count;
        self.sequence[item_index].is_1(bit_index % self.item_bits_count)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.bits_count());
        let item_index = bit_index / self.item_bits_count;
        self.sequence[item_index].flip_bit(bit_index % self.item_bits_count)
    }
}

impl<T, U> ByteChunkedBitBuffer for UniformSequence<T>
where
    T: std::ops::IndexMut<usize, Output = U>,
    for<'a> &'a T: IntoIterator<Item = &'a U>,
    U: ByteChunkedBitBuffer,
{
    fn bytes_count(&self) -> usize {
        let Some(first) = self.sequence.into_iter().next() else {
            debug_assert_eq!(self.items_count, 0);
            debug_assert_eq!(self.item_bits_count, 0);
            return 0;
        };

        first.bytes_count() * self.items_count
    }

    fn get_byte(&self, n: usize) -> u8 {
        debug_assert!(n < self.bytes_count());
        let item_index = (n * 8) / self.item_bits_count;
        let index_in_item = ((n * 8) % self.item_bits_count) / 8;
        self.sequence[item_index].get_byte(index_in_item)
    }

    fn set_byte(&mut self, n: usize, value: u8) {
        debug_assert!(n < self.bytes_count());
        let item_index = (n * 8) / self.item_bits_count;
        let index_in_item = ((n * 8) % self.item_bits_count) / 8;
        self.sequence[item_index].set_byte(index_in_item, value)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn creation_error() {
        let result = UniformSequence::new([vec![1, 2], vec![3, 4], vec![5, 6, 7], vec![8, 9, 10]]);
        assert_eq!(result, Err(NonMatchingIndex(2)));
    }

    #[test]
    fn empty() {
        let buffer: Vec<[u8; 6]> = vec![];
        let uniform = UniformSequence::new(buffer).unwrap();

        assert_eq!(uniform.items_count(), 0);
        assert_eq!(uniform.item_bits_count(), 0);
        assert_eq!(uniform.bytes_count(), 0);
        assert_eq!(uniform.bits_count(), 0);
    }

    #[test]
    fn common_ops() {
        let mut buf = UniformSequence::new([0u8, 0b1010u8]).unwrap();
        assert_eq!(buf.bits_count(), 16);
        assert_eq!(buf.item_bits_count(), 8);

        for i in 0..9 {
            assert!(buf.is_0(i));
        }
        assert!(buf.is_1(9));
        assert!(buf.is_0(10));
        assert!(buf.is_1(11));
        for i in 12..16 {
            assert!(buf.is_0(i));
        }

        buf.set_0(10);
        assert!(buf.is_0(10));
        buf.set_1(10);
        assert!(buf.is_1(10));

        buf.flip_bit(10);
        assert!(buf.is_0(10));
        buf.flip_bit(10);
        assert!(buf.is_1(10));

        assert_eq!(buf.bytes_count(), 2);
        assert_eq!(buf.get_byte(0), 0);
        assert_eq!(buf.get_byte(1), 0b1110);
        buf.set_byte(0, u8::MAX);
        assert_eq!(buf.get_byte(0), u8::MAX);

        let inner = buf.into_inner();
        assert_eq!(inner, [u8::MAX, 0b1110]);
    }

    #[test]
    fn multi_byte_items() {
        let mut buf = UniformSequence::new([0u32, 0xffu32]).unwrap();
        assert_eq!(buf.bytes_count(), 8);
        assert_eq!(buf.item_bits_count(), 32);

        assert_eq!(buf.get_byte(4), 0xff);
        buf.set_byte(1, 0xaa);
        assert_eq!(buf.get_byte(1), 0xaa);
        buf.set_byte(5, 0x88);
        assert_eq!(buf.get_byte(5), 0x88);
        buf.set_byte(4, 0xbb);
        assert_eq!(buf.get_byte(4), 0xbb);
    }
}
