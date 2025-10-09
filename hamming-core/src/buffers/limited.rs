use crate::prelude::*;

/// Limit the number of bits in a [`BitBuffer`].
pub struct Limited<T> {
    buffer: T,
    num_bits: usize,
}

impl<T> Limited<T>
where
    T: BitBuffer,
{
    /// Create a new limited bit buffer.
    ///
    /// # Panics
    ///
    /// if `num_bits` > `buffer.num_bits()`.
    pub fn new(buffer: T, num_bits: usize) -> Self {
        assert!(
            num_bits <= buffer.num_bits(),
            "the number of bits in the original buffer is {}, but tried to limit to {}",
            buffer.num_bits(),
            num_bits
        );
        Self { buffer, num_bits }
    }

    /// Extract the original bitbuffer.
    pub fn into_inner(self) -> T {
        self.buffer
    }
}

impl<T> BitBuffer for Limited<T>
where
    T: BitBuffer,
{
    fn num_bits(&self) -> usize {
        self.num_bits
    }

    fn is_1(&self, bit_index: usize) -> bool {
        if bit_index >= self.num_bits {
            panic!("{bit_index} is out of bounds");
        }
        self.buffer.is_1(bit_index)
    }

    fn set_0(&mut self, bit_index: usize) {
        if bit_index >= self.num_bits {
            panic!("{bit_index} is out of bounds");
        }
        self.buffer.set_0(bit_index)
    }

    fn set_1(&mut self, bit_index: usize) {
        if bit_index >= self.num_bits {
            panic!("{bit_index} is out of bounds");
        }
        self.buffer.set_1(bit_index)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        if bit_index >= self.num_bits {
            panic!("{bit_index} is out of bounds");
        }
        self.buffer.flip_bit(bit_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_bounds() {
        assert!(Limited::new(0b1001u8, 4).is_1(0));
        assert!(Limited::new(0b1001u8, 4).is_0(1));
        assert!(Limited::new(0b1001u8, 4).is_0(2));
        assert!(Limited::new(0b1001u8, 4).is_1(3));

        assert!(Limited::new(0b10000000u8, 8).is_0(6));
        assert!(Limited::new(0b10000000u8, 8).is_1(7));
    }

    #[test]
    #[should_panic]
    fn out_of_bounds() {
        Limited::new(0b0000, 4).is_1(4);
    }

    #[test]
    #[should_panic]
    fn invalid_num_bits() {
        Limited::new(0u8, 9);
    }

    #[test]
    fn bit_copy() {
        let num_bits = 15;
        let source = Limited::new([255u8; 2], num_bits);
        let mut dest = [0u8; 2];

        let copied = source.copy_into(0, &mut dest);
        assert_eq!(copied, num_bits);
        assert_eq!(dest, [255u8, 0b01111111u8]);

        let source = [255u8; 2];
        let mut dest = Limited::new([0u8; 2], num_bits);

        let copied = source.copy_into(0, &mut dest);
        assert_eq!(copied, num_bits);
        assert_eq!(dest.into_inner(), [255u8, 0b01111111u8]);
    }
}
