use super::BitBuffer;

/// An [`Iterator`] over the bits in a [`BitBuffer`].
///
/// `true` represents 1 and `false` 0.
pub struct Bits<'a, T: ?Sized> {
    buffer: &'a T,
    next_bit: usize,
}

impl<'a, T> Bits<'a, T>
where
    T: ?Sized,
{
    pub fn new(buffer: &'a T) -> Self {
        Self {
            buffer,
            next_bit: 0,
        }
    }
}

impl<'a, T> Iterator for Bits<'a, T>
where
    T: BitBuffer + ?Sized,
{
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        assert!(self.next_bit <= self.buffer.num_bits());
        if self.next_bit == self.buffer.num_bits() {
            return None;
        }

        let result = self.buffer.is_1(self.next_bit);
        self.next_bit += 1;
        Some(result)
    }
}
