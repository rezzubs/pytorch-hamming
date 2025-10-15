pub mod bits;
mod byte_chunked;
pub mod chunks;
mod random_picker;

pub use bits::Bits;
pub use byte_chunked::ByteChunkedBitBuffer;
use chunks::DynChunks;
use random_picker::RandomPicker;

use crate::{
    buffers::{limited::bytes_to_store_n_bits, Limited},
    encoding::{encode_into, num_encoded_bits},
};

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
    fn bits<'a>(&'a self) -> Bits<'a, Self> {
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

    /// Count the number of bits which are 1.
    fn num_1_bits(&self) -> usize {
        self.bits().filter(|is_1| *is_1).count()
    }

    /// Flip exactly n bits randomly in the buffer.
    ///
    /// All bit flips will be unique.
    ///
    /// # Panics
    ///
    /// - If `n > self.num_bits()`
    fn flip_n_bits(&mut self, n: usize) {
        let num_bits = self.num_bits();
        assert!(n <= num_bits);

        let mut possible_faults = RandomPicker::new(num_bits, rand::rng());

        for _ in 0..n {
            let fault_target = possible_faults.next().unwrap();
            self.flip_bit(fault_target);
        }
    }

    /// Flip a number of bits by the given bit error rate.
    ///
    /// All bit flips will be unique.
    ///
    /// Returns the number of bits flipped
    ///
    /// # Panics
    ///
    /// - if `ber` does not fit within `0..=1`.
    fn flip_by_ber(&mut self, ber: f64) -> usize {
        assert!((0f64..=1f64).contains(&ber));

        let num_flips = (self.num_bits() as f64 * ber) as usize;

        self.flip_n_bits(num_flips);

        num_flips
    }

    /// Copy all the bits from `self` to `other`
    ///
    /// Returns the number of bits copied.
    ///
    /// # Panics
    ///
    /// if `start >= self.num_bits()`.
    #[must_use]
    fn copy_into<O>(&self, start: usize, other: &mut O) -> usize
    where
        O: BitBuffer,
    {
        assert!(start < self.num_bits());

        for (source_i, dest_i) in (start..self.num_bits()).zip(0..other.num_bits()) {
            if self.is_1(source_i) {
                other.set_1(dest_i);
            } else {
                other.set_0(dest_i);
            }
        }

        // Either `self` was copied fully or was limited by the size of `other`.
        (self.num_bits() - start).min(other.num_bits())
    }

    /// Limit the number of perceived bits in the buffer.
    fn into_limited(self, num_bits: usize) -> Limited<Self>
    where
        Self: std::marker::Sized,
    {
        Limited::new(self, num_bits)
    }

    /// Convert the buffer to chunks of equal length.
    fn to_dyn_chunks(self, bits_per_chunk: usize) -> DynChunks
    where
        Self: Sized,
    {
        DynChunks::from_buffer(self, bits_per_chunk)
    }

    /// Encode the buffer as a hamming code.
    ///
    /// See [`encode_into`] for usage with custom output buffers.
    fn encode(&self) -> Limited<Vec<u8>>
    where
        Self: std::marker::Sized,
    {
        let num_encoded_bits = num_encoded_bits(self.num_bits());
        let num_bytes = bytes_to_store_n_bits(num_encoded_bits);
        let mut dest = vec![0u8; num_bytes].into_limited(num_encoded_bits);

        encode_into(self, &mut dest);

        dest
    }
}

/// A [`BitBuffer`] with a comptime known length.
pub trait SizedBitBuffer: BitBuffer {
    /// Total number of bits in the buffer.
    ///
    /// Must be an exact match with [`BitBuffer::num_bits`].
    const NUM_BITS: usize;
}

macro_rules! int_impl {
    ($t:ty, $bits:expr) => {
        impl SizedBitBuffer for $t {
            const NUM_BITS: usize = $bits;
        }

        impl BitBuffer for $t {
            fn num_bits(&self) -> usize {
                Self::NUM_BITS
            }

            fn set_1(&mut self, bit_index: usize) {
                debug_assert!(bit_index < Self::NUM_BITS, "{bit_index} is out of bounds");
                *self |= 1 << bit_index
            }

            fn set_0(&mut self, bit_index: usize) {
                debug_assert!(bit_index < Self::NUM_BITS, "{bit_index} is out of bounds");
                *self &= !(1 << bit_index)
            }

            fn is_1(&self, bit_index: usize) -> bool {
                debug_assert!(bit_index < Self::NUM_BITS, "{bit_index} is out of bounds");
                (self & (1 << bit_index)) > 0
            }

            fn flip_bit(&mut self, bit_index: usize) {
                debug_assert!(bit_index < Self::NUM_BITS, "{bit_index} is out of bounds");
                *self ^= 1 << bit_index
            }
        }
    };
}

int_impl!(u8, 8);
int_impl!(u16, 16);
int_impl!(u32, 32);
int_impl!(u64, 64);
int_impl!(u128, 128);
int_impl!(i8, 8);
int_impl!(i16, 16);
int_impl!(i32, 32);
int_impl!(i64, 64);
int_impl!(i128, 128);

macro_rules! float_impl {
    ($t:ty, $bits:expr) => {
        impl SizedBitBuffer for $t {
            const NUM_BITS: usize = $bits;
        }

        impl BitBuffer for $t {
            fn num_bits(&self) -> usize {
                Self::NUM_BITS
            }

            fn set_1(&mut self, bit_index: usize) {
                let mut unsigned = self.to_bits();
                unsigned.set_1(bit_index);
                *self = <$t>::from_bits(unsigned)
            }

            fn set_0(&mut self, bit_index: usize) {
                let mut unsigned = self.to_bits();
                unsigned.set_0(bit_index);
                *self = <$t>::from_bits(unsigned)
            }

            fn is_1(&self, bit_index: usize) -> bool {
                let unsigned = self.to_bits();
                unsigned.is_1(bit_index)
            }

            fn flip_bit(&mut self, bit_index: usize) {
                let mut unsigned = self.to_bits();
                unsigned.flip_bit(bit_index);
                *self = <$t>::from_bits(unsigned)
            }
        }
    };
}

float_impl!(f32, 32);
float_impl!(f64, 64);

impl<T> BitBuffer for [T]
where
    T: SizedBitBuffer,
{
    /// Total number of bits in the buffer.
    fn num_bits(&self) -> usize {
        self.len() * T::NUM_BITS
    }

    fn set_1(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.num_bits());
        let item_index = bit_index / T::NUM_BITS;
        self[item_index].set_1(bit_index % T::NUM_BITS);
    }

    fn set_0(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.num_bits());
        let item_index = bit_index / T::NUM_BITS;
        self[item_index].set_0(bit_index % T::NUM_BITS);
    }

    fn is_1(&self, bit_index: usize) -> bool {
        debug_assert!(bit_index < self.num_bits());
        let item_index = bit_index / T::NUM_BITS;
        self[item_index].is_1(bit_index % T::NUM_BITS)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        debug_assert!(bit_index < self.num_bits());
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

impl<T> BitBuffer for Vec<T>
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

#[cfg(test)]
mod tests {
    use super::*;

    mod u8 {
        use super::*;

        #[test]
        fn set1() {
            let mut val = 0b0001u8;
            val.set_1(0);
            assert_eq!(val, 0b0001);

            let mut val = 0b0001u8;
            val.set_1(1);
            assert_eq!(val, 0b0011);

            let mut val = 0b0001u8;
            val.set_1(2);
            assert_eq!(val, 0b0101);

            let mut val = 0b0001u8;
            val.set_1(3);
            assert_eq!(val, 0b1001);

            let mut val = 0b0000_0001u8;
            val.set_1(7);
            assert_eq!(val, 0b1000_0001);
        }

        #[test]
        fn set0() {
            let mut val = 0b1110u8;
            val.set_0(0);
            assert_eq!(val, 0b1110);

            let mut val = 0b1110u8;
            val.set_0(1);
            assert_eq!(val, 0b1100);

            let mut val = 0b1110u8;
            val.set_0(2);
            assert_eq!(val, 0b1010);

            let mut val = 0b1110u8;
            val.set_0(3);
            assert_eq!(val, 0b0110);

            let mut val = 0b1000_0001u8;
            val.set_0(7);
            assert_eq!(val, 0b0000_0001);
        }

        #[test]
        fn is_1() {
            assert!(0b0001u8.is_1(0));
            assert!(!0b0001u8.is_1(2));
            assert!(!0b0001u8.is_1(4));
            assert!(!0b0001u8.is_1(4));
            assert!(!0b0010u8.is_1(0));
            assert!(0b0010u8.is_1(1));
        }

        #[test]
        fn flip_bit() {
            let mut val = 0b0000u8;
            val.flip_bit(0);
            assert_eq!(val, 0b0001);

            let mut val = 0b0000u8;
            val.flip_bit(1);
            assert_eq!(val, 0b0010);

            let mut val = 0b0000u8;
            val.flip_bit(3);
            assert_eq!(val, 0b1000);

            let mut val = 0b1111u8;
            val.flip_bit(0);
            assert_eq!(val, 0b1110);
        }

        #[test]
        fn total_even() {
            assert!(0b00000000u8.total_parity_is_even());
            assert!(!0b00000001u8.total_parity_is_even());
            assert!(0b00000011u8.total_parity_is_even());
            assert!(!0b00000111u8.total_parity_is_even());
            assert!(0b10000001u8.total_parity_is_even());
            assert!(!0b10010001u8.total_parity_is_even());
            assert!(0b11111111u8.total_parity_is_even());
        }

        #[test]
        fn num_1_bits() {
            assert_eq!(0b00101100u8.num_1_bits(), 3);
            assert_eq!(0b1000001u8.num_1_bits(), 2);
        }

        #[test]
        fn fault_injection() {
            let mut buf = 0u8;
            buf.flip_n_bits(1);
            assert_eq!(buf.num_1_bits(), 1);

            let mut buf = 0u8;
            buf.flip_n_bits(2);
            assert_eq!(buf.num_1_bits(), 2);

            let mut buf = 0u8;
            buf.flip_n_bits(3);
            assert_eq!(buf.num_1_bits(), 3);

            let mut buf = 0u8;
            buf.flip_n_bits(4);
            assert_eq!(buf.num_1_bits(), 4);

            let mut buf = 0u8;
            buf.flip_by_ber(1.);
            assert_eq!(buf.num_1_bits(), 8);
        }
    }

    mod i32 {
        use super::*;

        #[test]
        fn flip() {
            let mut buf = 0i32;
            buf.flip_bit(31);
            assert_eq!(buf, -2147483648i32);
        }
    }

    mod f32 {
        use super::*;

        #[test]
        fn flip() {
            let mut buf = 1f32;
            buf.flip_bit(31);
            assert_eq!(buf, -1f32);
        }
    }

    mod f64 {
        use super::*;

        #[test]
        fn flip() {
            let mut buf = 1f64;
            buf.flip_bit(63);
            assert_eq!(buf, -1f64);
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

    #[test]
    fn copy_into_different_structure() {
        let a_actual: [u8; 8] = [123, 4, 255, 0, 2, 97, 34, 255];
        let num_bits = a_actual.len() * 8;
        let b_actual: [u16; 4] = [
            u16::from_le_bytes([123, 4]),
            u16::from_le_bytes([255, 0]),
            u16::from_le_bytes([2, 97]),
            u16::from_le_bytes([34, 255]),
        ];

        let mut b: [u16; 4] = [0xabc2, 0x1234, 0x1ab2, 0x4a89];
        let copied = a_actual.copy_into(0, &mut b);
        assert_eq!(copied, num_bits);
        assert_eq!(b, b_actual);

        let mut a: [u8; 8] = [0xfa, 0xab, 0x42, 0x01, 0xaa, 0x00, 0xff, 0x4c];
        let copied = b_actual.copy_into(0, &mut a);
        assert_eq!(copied, num_bits);
        assert_eq!(a, a_actual);
    }

    #[test]
    fn copy_into_partial() {
        let source = [255u8, 127u8, 63u8, 31u8];
        let mut dest = 0u8;

        let mut start = 0;
        for expected in source {
            let copied = source.copy_into(start, &mut dest);
            assert_eq!(copied, 8);
            start += copied;
            assert_eq!(dest, expected);
        }
    }
}
