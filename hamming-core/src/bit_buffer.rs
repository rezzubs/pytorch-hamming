pub trait BitBuffer {
    /// Number of bits stored by this buffer.
    fn num_bits(&self) -> usize;

    /// Set a bit with index `bit_index` to 1.
    fn set_1(&mut self, bit_index: usize);

    /// Set a bit with index `bit_index` to 0.
    fn set_0(&mut self, bit_index: usize);

    /// Check if the bit at index `bit_index` is 1.
    fn is_1(&self, bit_index: usize) -> bool;

    /// Check if the bit with index `bit_index` is 0.
    fn is_0(&self, bit_index: usize) -> bool {
        !self.is_1(bit_index)
    }

    /// Flip the bit with index `bit_index`.
    fn flip_bit(&mut self, bit_index: usize) {
        // NOTE: It's likely that a custom implementation for a specific type will be faster.
        if self.is_0(bit_index) {
            self.set_1(bit_index);
        } else {
            self.set_0(bit_index);
        }
    }

    /// Iterate over the bits of the array.
    fn bits(&self) -> Bits<Self> {
        Bits {
            buffer: self,
            next_bit: 0,
        }
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

/// A [`BitBuffer`] with a comptime known length.
pub trait SizedBitBuffer: BitBuffer {
    /// Total number of bits in the buffer.
    ///
    /// Must be an exact match with [`BitBuffer::num_bits`].
    const NUM_BITS: usize;
}

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

/// An [`Iterator`] over the bits in a [`BitBuffer`].
///
/// `true` represents 1 and `false` 0.
pub struct Bits<'a, T: ?Sized> {
    buffer: &'a T,
    next_bit: usize,
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

    mod u8 {
        use super::*;

        #[test]
        fn set1() {
            let mut val = 0b0001;
            val.set_1(0);
            assert_eq!(val, 0b0001);

            let mut val = 0b0001;
            val.set_1(1);
            assert_eq!(val, 0b0011);

            let mut val = 0b0001;
            val.set_1(2);
            assert_eq!(val, 0b0101);

            let mut val = 0b0001;
            val.set_1(3);
            assert_eq!(val, 0b1001);

            let mut val = 0b0000_0001;
            val.set_1(7);
            assert_eq!(val, 0b1000_0001);
        }

        #[test]
        fn set0() {
            let mut val = 0b1110;
            val.set_0(0);
            assert_eq!(val, 0b1110);

            let mut val = 0b1110;
            val.set_0(1);
            assert_eq!(val, 0b1100);

            let mut val = 0b1110;
            val.set_0(2);
            assert_eq!(val, 0b1010);

            let mut val = 0b1110;
            val.set_0(3);
            assert_eq!(val, 0b0110);

            let mut val = 0b1000_0001;
            val.set_0(7);
            assert_eq!(val, 0b0000_0001);
        }

        #[test]
        fn is_1() {
            assert!(0b0001.is_1(0));
            assert!(!0b0001.is_1(2));
            assert!(!0b0001.is_1(4));
            assert!(!0b0001.is_1(4));
            assert!(!0b0010.is_1(0));
            assert!(0b0010.is_1(1));
        }

        #[test]
        fn flip_bit() {
            let mut val = 0b0000;
            val.flip_bit(0);
            assert_eq!(val, 0b0001);

            let mut val = 0b0000;
            val.flip_bit(1);
            assert_eq!(val, 0b0010);

            let mut val = 0b0000;
            val.flip_bit(3);
            assert_eq!(val, 0b1000);

            let mut val = 0b1111;
            val.flip_bit(0);
            assert_eq!(val, 0b1110);
        }
    }

    mod sequence {
        use super::*;

        #[test]
        fn is_1() {
            let a = [0u8, 0b110u8];
            for i in 0..=8 {
                assert!(a.is_0(i))
            }
            assert!(a.is_1(9));
            assert!(a.is_1(10));
            assert!(a.is_0(11));
        }
    }
}
