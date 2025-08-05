// FIXME: remove after full implementation
#![allow(dead_code)]

/// An array of bytes with functions for manipulating single bits.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ByteArray<const N: usize> {
    data: [u8; N],
}

impl<const N: usize> ByteArray<N> {
    /// The number of bits stored in this array.
    pub const NUM_BITS: usize = N * 8;

    /// Query if the specified bit is set to high.
    ///
    /// # Panics
    ///
    /// This function panics when `bit_index` is out of bounds.
    pub fn bit_is_high(&self, bit_index: usize) -> bool {
        assert!(bit_index < Self::NUM_BITS);
        let byte_index = bit_index / 8;
        bit_is_high(self.data[byte_index], bit_index % 8)
    }

    /// Query if the specified bit is set to high.
    ///
    /// # Panics
    ///
    /// This function panics when `bit_index` is out of bounds.
    pub fn bit_is_low(&self, bit_index: usize) -> bool {
        assert!(bit_index < Self::NUM_BITS);
        let byte_index = bit_index / 8;
        bit_is_low(self.data[byte_index], bit_index % 8)
    }

    /// Iterate over the bits of the array.
    pub fn bits<'a>(&'a self) -> Bits<'a, N> {
        Bits {
            array: self,
            next_bit: 0,
        }
    }

    /// Iterate over the bytes of the array.
    pub fn into_bytes(self) -> std::array::IntoIter<u8, N> {
        self.data.into_iter()
    }

    /// Flip a bit in the array.
    pub fn flip_bit(&mut self, bit_index: usize) {
        assert!(bit_index < Self::NUM_BITS);
        let byte_index = bit_index / 8;
        self.data[byte_index] = flip_bit(self.data[byte_index], bit_index % 8);
    }

    /// Create a new bytearray with all bits set to zero.
    pub fn new() -> Self {
        Self { data: [0; N] }
    }

    /// Set the bit at at the specified index to 1.
    ///
    /// # Panics
    ///
    /// This function panics when `bit_index` is out of bounds.
    pub fn set_bit_high(&mut self, bit_index: usize) {
        assert!(bit_index < Self::NUM_BITS);
        let byte_index = bit_index / 8;
        self.data[byte_index] = set_bit_high(self.data[byte_index], bit_index % 8);
    }

    /// Set the bit at at the specified index to 0.
    ///
    /// # Panics
    ///
    /// This function panics when `bit_index` is out of bounds.
    pub fn set_bit_low(&mut self, bit_index: usize) {
        assert!(bit_index < Self::NUM_BITS);
        let byte_index = bit_index / 8;
        self.data[byte_index] = set_bit_low(self.data[byte_index], bit_index % 8);
    }
}

impl<const N: usize> Default for ByteArray<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl From<u64> for ByteArray<8> {
    fn from(value: u64) -> Self {
        Self::from(value.to_le_bytes())
    }
}

impl From<[f32; 2]> for ByteArray<8> {
    fn from(value: [f32; 2]) -> Self {
        let a_bytes = value[0].to_le_bytes();
        let b_bytes = value[1].to_le_bytes();
        let bytes = [
            a_bytes[0], a_bytes[1], a_bytes[2], a_bytes[3], b_bytes[0], b_bytes[1], b_bytes[2],
            b_bytes[3],
        ];
        Self::from(bytes)
    }
}

impl From<ByteArray<8>> for [f32; 2] {
    fn from(value: ByteArray<8>) -> Self {
        let bytes = value.data;
        let a_bytes = [bytes[0], bytes[1], bytes[2], bytes[3]];
        let b_bytes = [bytes[4], bytes[5], bytes[6], bytes[7]];

        [f32::from_le_bytes(a_bytes), f32::from_le_bytes(b_bytes)]
    }
}

impl From<u8> for ByteArray<1> {
    fn from(value: u8) -> Self {
        Self::from(value.to_le_bytes())
    }
}

impl<const N: usize> From<[u8; N]> for ByteArray<N> {
    fn from(value: [u8; N]) -> Self {
        Self { data: value }
    }
}

impl<const N: usize> std::fmt::Display for ByteArray<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "0b{}",
            self.bits()
                .map(|bit| if bit { '1' } else { '0' })
                // FIXME: implement double ended iteration for Bits.
                .collect::<Vec<char>>()
                .into_iter()
                .rev()
                .collect::<String>()
        )
    }
}

pub struct Bits<'a, const N: usize> {
    array: &'a ByteArray<N>,
    next_bit: usize,
}

impl<'a, const N: usize> Iterator for Bits<'a, N> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        assert!(self.next_bit <= N * 8);
        if self.next_bit == N * 8 {
            return None;
        }

        let result = self.array.bit_is_high(self.next_bit);
        self.next_bit += 1;
        Some(result)
    }
}

/// Set a bit of a byte to 1.
///
/// # Panics
///
/// This function panics when `bit_index` is out of bounds.
pub fn set_bit_high(byte: u8, bit_index: usize) -> u8 {
    assert!(bit_index <= 7);
    byte | (1 << bit_index)
}

/// Set a bit of a byte to 0.
///
/// # Panics
///
/// This function panics when `bit_index` is out of bounds.
pub fn set_bit_low(byte: u8, bit_index: usize) -> u8 {
    assert!(bit_index <= 7);
    byte & (!(1 << bit_index))
}

/// Queries if the specified bit is set low.
///
/// # Panics
///
/// This function panics when `bit_index` is out of bounds.
pub fn bit_is_high(byte: u8, bit_index: usize) -> bool {
    assert!(bit_index <= 7);
    (byte & (1 << bit_index)) > 0
}

/// Queries if the specified bit is set low.
///
/// # Panics
///
/// This function panics when `bit_index` is out of bounds.
pub fn bit_is_low(byte: u8, bit_index: usize) -> bool {
    !bit_is_high(byte, bit_index)
}

pub fn flip_bit(byte: u8, bit_index: usize) -> u8 {
    assert!(bit_index <= 7);
    byte ^ (1 << bit_index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set1_test() {
        assert_eq!(set_bit_high(0b0001, 0), 0b0001);
        assert_eq!(set_bit_high(0b0001, 1), 0b0011);
        assert_eq!(set_bit_high(0b0001, 2), 0b0101);
        assert_eq!(set_bit_high(0b0001, 3), 0b1001);
        assert_eq!(set_bit_high(0b0000_0001, 7), 0b1000_0001);
    }

    #[test]
    fn set0_test() {
        assert_eq!(set_bit_low(0b1110, 0), 0b1110);
        assert_eq!(set_bit_low(0b1110, 1), 0b1100);
        assert_eq!(set_bit_low(0b1110, 2), 0b1010);
        assert_eq!(set_bit_low(0b1110, 3), 0b0110);
        assert_eq!(set_bit_low(0b1000_0001, 7), 0b0000_0001);
    }

    #[test]
    fn get_bit_test() {
        assert_eq!(bit_is_high(0b0001, 0), true);
        assert_eq!(bit_is_high(0b0001, 2), false);
        assert_eq!(bit_is_high(0b0001, 4), false);
        assert_eq!(bit_is_high(0b0001, 4), false);
        assert_eq!(bit_is_high(0b0010, 0), false);
        assert_eq!(bit_is_high(0b0010, 1), true);
    }

    #[test]
    fn flip_test() {
        assert_eq!(flip_bit(0b0000, 0), 0b0001);
        assert_eq!(flip_bit(0b0000, 1), 0b0010);
        assert_eq!(flip_bit(0b0000, 3), 0b1000);
        assert_eq!(flip_bit(0b1111, 0), 0b1110);
    }

    fn assert_f32(arr: [f32; 2]) {
        let initial = arr;
        let converted = ByteArray::from(initial);
        let after: [f32; 2] = converted.into();
        assert_eq!(initial, after);
    }

    #[test]
    fn f32_conversions() {
        assert_f32([42.0, 6.9]);
        assert_f32([0.9937, 0.2209]);
    }
}
