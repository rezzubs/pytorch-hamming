use crate::{BitBuffer, SizedBitBuffer};

/// A [`BitBuffer`] with padding in regular intervals.
///
/// `D` marks the number of data bits.
/// `P` marks the number of padding bits. These will be ignored by bit operations.
///
/// `D` and `P` are going to distribute evenly in the original buffer. Here's an example with a
/// `BitBuffer<u8, 3, 1>`, using `D` for data and `P` for padding (most significant bit on the left
/// side)
/// ```text
/// original u8:  0bDDDDDDDD
/// PaddedBuffer: 0bPDDDPDDD
/// ```
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default, Hash)]
pub struct PaddedBuffer<T, const D: usize, const P: usize>(T);

impl<T, const D: usize, const P: usize> PaddedBuffer<T, D, P> {
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T, const D: usize, const P: usize> PaddedBuffer<T, D, P>
where
    T: BitBuffer,
{
    /// The number of bits in a single data + padding container.
    const CONTAINER_BITS: usize = D + P;

    /// Create a new padded buffer.
    ///
    /// # Panics
    ///
    /// - If `original` is not a multiple of `D + P`
    pub fn new(original: T) -> PaddedBuffer<T, D, P> {
        assert_eq!(original.num_bits() % Self::CONTAINER_BITS, 0);
        PaddedBuffer(original)
    }

    /// Return the number of data + padding containers.
    pub fn num_containers(&self) -> usize {
        self.0.num_bits() / Self::CONTAINER_BITS
    }

    /// Compute the index into the internal buffer.
    fn true_index(&self, index: usize) -> usize {
        assert!(index < self.num_bits(), "out of bounds");

        let containers_before_bit = index / D;
        let container_start = containers_before_bit * Self::CONTAINER_BITS;

        container_start + (index % D)
    }
}

impl<T, const D: usize, const P: usize> BitBuffer for PaddedBuffer<T, D, P>
where
    T: BitBuffer,
{
    fn num_bits(&self) -> usize {
        self.num_containers() * D
    }

    fn set_1(&mut self, bit_index: usize) {
        self.0.set_1(self.true_index(bit_index));
    }

    fn set_0(&mut self, bit_index: usize) {
        self.0.set_0(self.true_index(bit_index));
    }

    fn is_1(&self, bit_index: usize) -> bool {
        self.0.is_1(self.true_index(bit_index))
    }

    fn flip_bit(&mut self, bit_index: usize) {
        self.0.flip_bit(self.true_index(bit_index));
    }
}

impl<T, const D: usize, const P: usize> SizedBitBuffer for PaddedBuffer<T, D, P>
where
    T: SizedBitBuffer,
{
    const NUM_BITS: usize = (T::NUM_BITS / Self::CONTAINER_BITS) * D;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_1() {
        type A = PaddedBuffer<u8, 1, 1>;

        let buf = A::new(0b00010001);
        assert_eq!(buf.num_bits(), 4);
        assert!(buf.is_1(0));
        assert!(buf.is_0(1));
        assert!(buf.is_1(2));
        assert!(buf.is_0(3));

        let buf = A::new(0b01000100);
        assert!(buf.is_0(0));
        assert!(buf.is_1(1));
        assert!(buf.is_0(2));
        assert!(buf.is_1(3));

        let buf = A::new(0b10101010);
        assert!(buf.is_0(0));
        assert!(buf.is_0(1));
        assert!(buf.is_0(2));
        assert!(buf.is_0(3));

        type B = PaddedBuffer<u8, 1, 3>;

        let buf = B::new(0b11101110);
        assert_eq!(buf.num_bits(), 2);
        assert!(buf.is_0(0));
        assert!(buf.is_0(1));

        let buf = B::new(0b00010000);
        assert_eq!(buf.num_bits(), 2);
        assert!(buf.is_0(0));
        assert!(buf.is_1(1));

        type C = PaddedBuffer<u8, 3, 1>;

        let buf = C::new(0b00010011);
        assert_eq!(buf.num_bits(), 6);
        assert!(buf.is_1(0));
        assert!(buf.is_1(1));
        assert!(buf.is_0(2));
        assert!(buf.is_1(3));
        assert!(buf.is_0(4));
        assert!(buf.is_0(5));
    }

    #[test]
    fn num_bits() {
        type T = PaddedBuffer<[u8; 2], 3, 5>;
        let buf = T::new(Default::default());
        assert_eq!(T::NUM_BITS, buf.num_bits())
    }
}
