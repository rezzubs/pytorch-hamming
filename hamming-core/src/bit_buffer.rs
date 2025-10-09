mod float;
mod int;
mod random_picker;
mod sequence;
#[cfg(test)]
mod tests;

use random_picker::RandomPicker;

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
}

/// A [`BitBuffer`] with a comptime known length.
pub trait SizedBitBuffer: BitBuffer {
    /// Total number of bits in the buffer.
    ///
    /// Must be an exact match with [`BitBuffer::num_bits`].
    const NUM_BITS: usize;
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
