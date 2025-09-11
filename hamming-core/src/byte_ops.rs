//! Functions for manipulating bits in bytes.

/// Set a bit of a byte to 1.
///
/// # Panics
///
/// This function panics when `bit_index` is out of bounds.
pub fn set_1(byte: u8, bit_index: usize) -> u8 {
    assert!(bit_index <= 7);
    byte | (1 << bit_index)
}

/// Set a bit of a byte to 0.
///
/// # Panics
///
/// This function panics when `bit_index` is out of bounds.
pub fn set_0(byte: u8, bit_index: usize) -> u8 {
    assert!(bit_index <= 7);
    byte & (!(1 << bit_index))
}

/// Queries if the specified bit is set low.
///
/// # Panics
///
/// This function panics when `bit_index` is out of bounds.
pub fn is_1(byte: u8, bit_index: usize) -> bool {
    assert!(bit_index <= 7);
    (byte & (1 << bit_index)) > 0
}

/// Queries if the specified bit is set low.
///
/// # Panics
///
/// This function panics when `bit_index` is out of bounds.
pub fn is_0(byte: u8, bit_index: usize) -> bool {
    !is_1(byte, bit_index)
}

pub fn flip(byte: u8, bit_index: usize) -> u8 {
    assert!(bit_index <= 7);
    byte ^ (1 << bit_index)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set1_test() {
        assert_eq!(set_1(0b0001, 0), 0b0001);
        assert_eq!(set_1(0b0001, 1), 0b0011);
        assert_eq!(set_1(0b0001, 2), 0b0101);
        assert_eq!(set_1(0b0001, 3), 0b1001);
        assert_eq!(set_1(0b0000_0001, 7), 0b1000_0001);
    }

    #[test]
    fn set0_test() {
        assert_eq!(set_0(0b1110, 0), 0b1110);
        assert_eq!(set_0(0b1110, 1), 0b1100);
        assert_eq!(set_0(0b1110, 2), 0b1010);
        assert_eq!(set_0(0b1110, 3), 0b0110);
        assert_eq!(set_0(0b1000_0001, 7), 0b0000_0001);
    }

    #[test]
    fn get_bit_test() {
        assert!(is_1(0b0001, 0));
        assert!(!is_1(0b0001, 2));
        assert!(!is_1(0b0001, 4));
        assert!(!is_1(0b0001, 4));
        assert!(!is_1(0b0010, 0));
        assert!(is_1(0b0010, 1));
    }

    #[test]
    fn flip_test() {
        assert_eq!(flip(0b0000, 0), 0b0001);
        assert_eq!(flip(0b0000, 1), 0b0010);
        assert_eq!(flip(0b0000, 3), 0b1000);
        assert_eq!(flip(0b1111, 0), 0b1110);
    }
}
