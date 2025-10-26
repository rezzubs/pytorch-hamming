use crate::{
    bit_buffer::chunks::{ByteChunks, Chunks},
    prelude::*,
};

/// A [`BitBuffer`] that is byte addressable.
pub trait ByteChunkedBitBuffer: BitBuffer {
    /// Return the number of bytes in the buffer.
    fn num_bytes(&self) -> usize;

    /// Get the byte at index `n`.
    fn get_byte(&self, n: usize) -> u8;

    /// Set byte `n` to `value`.
    fn set_byte(&mut self, n: usize, value: u8);

    /// Copy all the bytes from `self` to `other`
    ///
    /// Returns the number of bytes copied.
    ///
    /// # Panics
    ///
    /// if `start >= self.num_bytes()`.
    #[must_use]
    fn copy_into_chunked<O>(&self, start: usize, other: &mut O) -> usize
    where
        O: ByteChunkedBitBuffer,
    {
        assert!(start < self.num_bytes());

        for (source_i, dest_i) in (start..self.num_bytes()).zip(0..other.num_bytes()) {
            let byte = self.get_byte(source_i);

            other.set_byte(dest_i, byte);
        }

        // Either `self` was copied fully or was limited by the size of `other`.
        (self.num_bytes() - start).min(other.num_bytes())
    }

    /// Convert to chunks of equal length where each chunk is a number of bytes long.
    ///
    /// If the number of bytes in the buffer isn't a multiple of the number of bytes per chunk then
    /// it will result in a number of bytes of (essentially useless) padding at the end of the final
    /// chunk.
    fn to_byte_chunks(&self, bytes_per_chunk: usize) -> ByteChunks
    where
        Self: std::marker::Sized,
    {
        ByteChunks::from_buffer(self, bytes_per_chunk)
    }

    /// Convert to chunks of equal length.
    ///
    /// Choose the optimal format automatically. For manual selection see:
    /// - [`ByteChunkedBitBuffer::to_byte_chunks`]
    /// - [`BitBuffer::to_dyn_chunks`]
    ///
    /// If the number of bits in the buffer isn't a multiple of the number of bits per chunk then
    /// it will result in a number of bits of (essentially useless) padding at the end of the final
    /// chunk.
    fn to_chunks(&self, bits_per_chunk: usize) -> Chunks
    where
        Self: std::marker::Sized,
    {
        Chunks::from_buffer(self, bits_per_chunk)
    }
}

impl ByteChunkedBitBuffer for u8 {
    fn num_bytes(&self) -> usize {
        1
    }

    fn get_byte(&self, n: usize) -> u8 {
        assert_eq!(n, 0, "u8 only has one byte, got index {n}");
        *self
    }

    fn set_byte(&mut self, n: usize, value: u8) {
        assert_eq!(n, 0, "u8 only has one byte, got index {n}");
        *self = value;
    }
}

macro_rules! uint_impl {
    ($ty:ty, $bytes:expr) => {
        impl ByteChunkedBitBuffer for $ty {
            fn num_bytes(&self) -> usize {
                $bytes
            }

            fn get_byte(&self, n: usize) -> u8 {
                assert!(n < $bytes, "buffer has {} bytes, got index {}", $bytes, n);

                (self >> (n * 8)) as u8
            }

            fn set_byte(&mut self, n: usize, value: u8) {
                let num_bits = n * 8;
                let value_shifted = (value as $ty) << (num_bits);
                let mask: $ty = !(0xff << num_bits);

                *self &= mask;
                *self |= value_shifted;
            }
        }
    };
}

uint_impl!(u16, 2);
uint_impl!(u32, 4);
uint_impl!(u64, 8);

macro_rules! float_impl {
    ($ty:ty, $bytes:expr) => {
        impl ByteChunkedBitBuffer for $ty {
            fn num_bytes(&self) -> usize {
                $bytes
            }

            fn get_byte(&self, n: usize) -> u8 {
                self.to_bits().get_byte(n)
            }

            fn set_byte(&mut self, n: usize, value: u8) {
                let mut uint = self.to_bits();
                uint.set_byte(n, value);
                *self = <$ty>::from_bits(uint);
            }
        }
    };
}

float_impl!(f32, 4);
float_impl!(f64, 8);

impl<T> ByteChunkedBitBuffer for [T]
where
    T: ByteChunkedBitBuffer + SizedBitBuffer,
{
    fn num_bytes(&self) -> usize {
        self.len() * (T::NUM_BITS / 8)
    }

    fn get_byte(&self, n: usize) -> u8 {
        let child_num_bytes = T::NUM_BITS / 8;
        let item_index = n / child_num_bytes;
        let index_in_item = n % child_num_bytes;
        let item = self.get(item_index).unwrap_or_else(|| panic!("out of bounds, child_num_bytes: {child_num_bytes}, item_index: {item_index}, index_in_item: {index_in_item}"));
        item.get_byte(index_in_item)
    }

    fn set_byte(&mut self, n: usize, value: u8) {
        let num_bytes = T::NUM_BITS / 8;
        let item_index = n / num_bytes;
        let index_in_item = n % num_bytes;
        let Some(item) = self.get_mut(item_index) else {
            panic!("{n} is out of bounds");
        };
        item.set_byte(index_in_item, value)
    }
}

impl<const N: usize, T> ByteChunkedBitBuffer for [T; N]
where
    T: ByteChunkedBitBuffer + SizedBitBuffer,
{
    fn num_bytes(&self) -> usize {
        self.as_slice().num_bytes()
    }

    fn get_byte(&self, n: usize) -> u8 {
        self.as_slice().get_byte(n)
    }

    fn set_byte(&mut self, n: usize, value: u8) {
        self.as_mut_slice().set_byte(n, value)
    }
}

impl<T> ByteChunkedBitBuffer for Vec<T>
where
    T: ByteChunkedBitBuffer + SizedBitBuffer,
{
    fn num_bytes(&self) -> usize {
        self.as_slice().num_bytes()
    }

    fn get_byte(&self, n: usize) -> u8 {
        self.as_slice().get_byte(n)
    }

    fn set_byte(&mut self, n: usize, value: u8) {
        self.as_mut_slice().set_byte(n, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_byte() {
        let source = [450u16, 12u16, 0];
        assert_eq!(source.get_byte(0), source[0].to_le_bytes()[0]);
        assert_eq!(source.get_byte(1), source[0].to_le_bytes()[1]);
        assert_eq!(source.get_byte(2), source[1].to_le_bytes()[0]);
        assert_eq!(source.get_byte(3), source[1].to_le_bytes()[1]);
        assert_eq!(source.get_byte(4), source[2].to_le_bytes()[0]);
        assert_eq!(source.get_byte(5), source[2].to_le_bytes()[1]);
    }

    #[test]
    fn set_byte() {
        let source = [450u16, 12u16, 0];
        let mut dest = [0u16, 0u16, 258u16];
        for i in 0..6 {
            dest.set_byte(i, source.get_byte(i));
        }
        assert_eq!(dest.get_byte(0), source[0].to_le_bytes()[0]);
        assert_eq!(dest.get_byte(1), source[0].to_le_bytes()[1]);
        assert_eq!(dest.get_byte(2), source[1].to_le_bytes()[0]);
        assert_eq!(dest.get_byte(3), source[1].to_le_bytes()[1]);
        assert_eq!(dest.get_byte(4), source[2].to_le_bytes()[0]);
        assert_eq!(dest.get_byte(5), source[2].to_le_bytes()[1]);
    }

    #[test]
    fn copy_into_different_structure() {
        let a_actual: Vec<u8> = vec![123, 4, 255, 0, 2, 97, 34, 255];
        let num_bytes = a_actual.len();
        let b_actual: Vec<u16> = vec![
            u16::from_le_bytes([123, 4]),
            u16::from_le_bytes([255, 0]),
            u16::from_le_bytes([2, 97]),
            u16::from_le_bytes([34, 255]),
        ];

        let mut b = vec![0u16; 4];
        let copied = a_actual.copy_into_chunked(0, &mut b);
        assert_eq!(copied, num_bytes);
        assert_eq!(b, b_actual);

        let mut a = vec![0u8; 8];
        let copied = b_actual.copy_into_chunked(0, &mut a);
        assert_eq!(copied, num_bytes);
        assert_eq!(a, a_actual);
    }

    #[test]
    fn copy_into_partial() {
        let source = [255u8, 127u8, 63u8, 31u8];
        let mut dest = 0b01011010u8;

        let mut start = 0;
        for expected in source {
            let copied = source.copy_into_chunked(start, &mut dest);
            assert_eq!(copied, 1);
            start += copied;
            assert_eq!(dest, expected);
        }

        let mut dest = [0b01010101, 0b10101010];

        let mut start = 0;
        for expected in source.chunks(2) {
            let copied = source.copy_into_chunked(start, &mut dest);
            assert_eq!(copied, 2);
            start += copied;
            assert_eq!(dest, expected);
        }
    }

    #[test]
    fn uint_impl() {
        let source = 123456u32;
        let mut dest = 0u32;

        for (i, byte) in source.to_le_bytes().into_iter().enumerate() {
            assert_eq!(source.get_byte(i), byte);
            dest.set_byte(i, byte);
        }

        assert_eq!(source, dest);

        let source = 123456743032483000u64;
        let mut dest = 0u64;

        for (i, byte) in source.to_le_bytes().into_iter().enumerate() {
            assert_eq!(source.get_byte(i), byte);
            dest.set_byte(i, byte);
        }

        assert_eq!(source, dest);

        let source = 10456u16;
        let mut dest = 0u16;

        for (i, byte) in source.to_le_bytes().into_iter().enumerate() {
            assert_eq!(source.get_byte(i), byte);
            dest.set_byte(i, byte);
        }

        assert_eq!(source, dest);
    }

    #[test]
    fn float_impl() {
        let source = 13456.029f32;
        let mut dest = 0f32;

        for (i, byte) in source.to_le_bytes().into_iter().enumerate() {
            assert_eq!(source.get_byte(i), byte);
            dest.set_byte(i, byte);
        }

        assert_eq!(source, dest);

        let source = 12345324789.483002f64;
        let mut dest = 0f64;

        for (i, byte) in source.to_le_bytes().into_iter().enumerate() {
            assert_eq!(source.get_byte(i), byte);
            dest.set_byte(i, byte);
        }

        assert_eq!(source, dest);
    }

    mod out_of_bounds {
        use super::*;

        #[test]
        #[should_panic]
        fn u8_get() {
            0u8.get_byte(1);
        }

        #[test]
        #[should_panic]
        fn u16_get() {
            0u16.get_byte(2);
        }

        #[test]
        #[should_panic]
        fn u32_get() {
            0u32.get_byte(4);
        }

        #[test]
        #[should_panic]
        fn u64_get() {
            0u64.get_byte(8);
        }

        #[test]
        #[should_panic]
        fn u8_set() {
            0u8.set_byte(1, 0);
        }

        #[test]
        #[should_panic]
        fn u16_set() {
            0u16.set_byte(2, 0);
        }

        #[test]
        #[should_panic]
        fn u32_set() {
            0u32.set_byte(4, 0);
        }

        #[test]
        #[should_panic]
        fn u64_set() {
            0u64.set_byte(8, 0);
        }

        #[test]
        #[should_panic]
        fn f32_get() {
            0f32.get_byte(4);
        }

        #[test]
        #[should_panic]
        fn f64_get() {
            0f64.get_byte(8);
        }

        #[test]
        #[should_panic]
        fn f32_set() {
            0f32.set_byte(4, 0);
        }

        #[test]
        #[should_panic]
        fn f64_set() {
            0f64.set_byte(8, 0);
        }
    }
}
