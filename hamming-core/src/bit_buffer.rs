use crate::byte_ops::{flip, is_1, set_0, set_1};

pub trait BitBuffer: Sized {
    /// Number of bits stored by this buffer.
    fn num_bits(&self) -> usize;

    /// Set a bit with index `bit_idx` to 1.
    fn set_1(&mut self, bit_idx: usize);

    /// Set a bit with index `bit_idx` to 0.
    fn set_0(&mut self, bit_idx: usize);

    /// Check if the bit at index `bit_idx` is 1.
    fn is_1(&self, bit_idx: usize) -> bool;

    /// Check if the bit with index `bit_idx` is 0.
    fn is_0(&self, bit_idx: usize) -> bool {
        !self.is_1(bit_idx)
    }

    /// Flip the bit with index `bit_idx`.
    fn flip_bit(&mut self, bit_idx: usize) {
        // NOTE: It's likely that a custom implementation for a specific type will be faster.
        if self.is_0(bit_idx) {
            self.set_1(bit_idx);
        } else {
            self.set_0(bit_idx);
        }
    }

    /// Iterate over the bits of the array.
    fn bits(&self) -> Bits<Self> {
        Bits::new(self)
    }

    /// Return true if the number 1 bits is even.
    fn total_parity_is_even(&self) -> bool {
        self.bits().filter(|is_1| *is_1).count() % 2 == 0
    }

    /// Return a string of the bit representation.
    ///
    /// For example, `5u8` would become `0b00000101`.
    fn bit_string(&self) -> String {
        let bits = self
            .bits()
            .map(|bit| if bit { '1' } else { '0' })
            // FIXME: implement double ended iteration for Bits to remove the collect + rev.
            .collect::<Vec<char>>()
            .into_iter()
            .rev();

        "0b".chars().chain(bits).collect()
    }
}

impl<const N: usize> BitBuffer for [u8; N] {
    fn num_bits(&self) -> usize {
        N * 8
    }

    fn set_1(&mut self, bit_idx: usize) {
        assert!(bit_idx < self.num_bits());
        let byte_idx = bit_idx / 8;
        self[byte_idx] = set_1(self[byte_idx], bit_idx % 8);
    }

    fn set_0(&mut self, bit_idx: usize) {
        assert!(bit_idx < self.num_bits());
        let byte_idx = bit_idx / 8;
        self[byte_idx] = set_0(self[byte_idx], bit_idx % 8);
    }

    fn is_1(&self, bit_idx: usize) -> bool {
        assert!(bit_idx < self.num_bits());
        let byte_idx = bit_idx / 8;
        is_1(self[byte_idx], bit_idx % 8)
    }

    fn flip_bit(&mut self, bit_idx: usize) {
        assert!(bit_idx < self.num_bits());
        let byte_idx = bit_idx / 8;
        self[byte_idx] = flip(self[byte_idx], bit_idx % 8);
    }
}

/// An [`Iterator`] over the bits in a [`BitBuffer`].
///
/// `true` represents 1 and `false` 0.
pub struct Bits<'a, T> {
    buffer: &'a T,
    next_bit: usize,
}

impl<'a, T> Bits<'a, T> {
    pub fn new(buffer: &'a T) -> Self {
        Self {
            buffer,
            next_bit: 0,
        }
    }
}

impl<'a, T> Iterator for Bits<'a, T>
where
    T: BitBuffer,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_even() {
        assert!([0b00000000u8].total_parity_is_even());
        assert!(![0b00000001u8].total_parity_is_even());
        assert!([0b00000011u8].total_parity_is_even());
        assert!(![0b00000111u8].total_parity_is_even());
        assert!([0b10000001u8].total_parity_is_even());
        assert!(![0b10010001u8].total_parity_is_even());
        assert!([0b11111111u8].total_parity_is_even());
    }
}
