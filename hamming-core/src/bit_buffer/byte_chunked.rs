use crate::{bit_buffer::chunks::ByteChunks, prelude::*};

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

    fn to_byte_chunks(self, bytes_per_chunk: usize) -> ByteChunks
    where
        Self: std::marker::Sized,
    {
        ByteChunks::from_buffer(self, bytes_per_chunk)
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
    ($ty:ty, $into:ty, $bytes:expr) => {
        impl ByteChunkedBitBuffer for $ty {
            fn num_bytes(&self) -> usize {
                $bytes
            }

            fn get_byte(&self, n: usize) -> u8 {
                (*self as $into).get_byte(n)
            }

            fn set_byte(&mut self, n: usize, value: u8) {
                let mut uint = *self as $into;
                uint.set_byte(n, value);
                *self = uint as $ty;
            }
        }
    };
}

float_impl!(f32, u32, 4);
float_impl!(f64, u64, 8);

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
}
