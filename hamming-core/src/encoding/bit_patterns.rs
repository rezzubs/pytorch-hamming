#![allow(dead_code)]

use std::collections::HashSet;

use crate::{
    bit_buffer::chunks::{self, Chunks, DynChunks},
    buffers::Limited,
    prelude::*,
};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct BitPattern {
    mask: HashSet<usize>,
    length: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
/// The length of the buffer isn't a multiple of the [`BitPattern`].
#[error("The length of the buffer isn't a multiple of the bit pattern")]
pub struct LengthMismatch;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
pub enum CreationError {
    #[error("Length too small, expected at least: {expected_at_least}")]
    InvalidLength { expected_at_least: usize },
    #[error("This pattern would protect all bits. Some other BitBuffer should be used instead.")]
    AllProtected,
    #[error("These parameters would result in an empty pattern")]
    Empty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BufferParams {
    num_total_bits: usize,
    num_masks: usize,
    num_protected_bits: usize,
}

impl BufferParams {
    fn num_unprotected_bits(&self) -> usize {
        self.num_total_bits - self.num_protected_bits
    }
}

impl BitPattern {
    pub fn new(
        mask: impl IntoIterator<Item = usize>,
        length: usize,
    ) -> Result<Self, CreationError> {
        if length == 0 {
            return Err(CreationError::Empty);
        }

        let mask: HashSet<usize> = mask.into_iter().collect();

        let Some(&max) = mask.iter().max() else {
            return Err(CreationError::Empty);
        };

        if length <= max {
            return Err(CreationError::InvalidLength {
                expected_at_least: max + 1,
            });
        }

        if length == mask.len() {
            return Err(CreationError::AllProtected);
        }

        Ok(Self { mask, length })
    }

    #[must_use]
    pub fn mask(&self) -> &HashSet<usize> {
        &self.mask
    }

    #[must_use]
    pub fn length(&self) -> usize {
        self.length
    }
}

impl BitPattern {
    fn buffer_params<B>(&self, buffer: &B) -> Result<BufferParams, LengthMismatch>
    where
        B: BitBuffer,
    {
        assert!(self.length >= self.mask.len());

        let num_total_bits = buffer.num_bits();
        if !num_total_bits.is_multiple_of(self.length) {
            return Err(LengthMismatch);
        }

        let num_masks = buffer.num_bits() / self.length;
        // NOTE: the length of the mask corresponds to the number of bits to be encoded per pattern.
        let num_protected_bits = self.mask.len() * num_masks;
        assert!(num_protected_bits <= num_total_bits);

        Ok(BufferParams {
            num_total_bits,
            num_masks,
            num_protected_bits,
        })
    }

    pub fn partition<B>(
        &self,
        buffer: &B,
        bits_per_chunk: usize,
    ) -> Result<(Limited<Vec<u8>>, Chunks), LengthMismatch>
    where
        B: BitBuffer,
    {
        let params = self.buffer_params(buffer)?;

        let mut protected = Chunks::zero(params.num_protected_bits, bits_per_chunk);
        let mut unprotected = Limited::bytes(params.num_unprotected_bits());

        let mut protected_i = 0;
        let mut unprotected_i = 0;
        for (i, is_1) in buffer.bits().enumerate() {
            let mask_i = i % self.length;

            if self.mask.contains(&mask_i) {
                if is_1 {
                    protected.set_1(protected_i);
                }
                protected_i += 1;
            } else {
                if is_1 {
                    unprotected.set_1(unprotected_i);
                }
                unprotected_i += 1;
            }
        }

        Ok((unprotected, protected))
    }

    pub fn encode<B>(
        &self,
        buffer: &B,
        bits_per_chunk: usize,
    ) -> Result<BitPatternEncoding, LengthMismatch>
    where
        B: BitBuffer,
    {
        let (unprotected, protected) = self.partition(buffer, bits_per_chunk)?;

        Ok(BitPatternEncoding {
            data: BitPatternEncodingData {
                protected: protected.encode_chunks(),
                unprotected,
            },
            bits_per_chunk,
            pattern: self.clone(),
        })
    }
}

/// The raw bytes that are made up of the orgiginal data bits + parity bits.
///
/// A bit buffer implementation is also provided. The unprotected bits come before the protected
/// bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitPatternEncodingData {
    pub(crate) unprotected: Limited<Vec<u8>>,
    pub(crate) protected: DynChunks,
}

impl BitBuffer for BitPatternEncodingData {
    fn num_bits(&self) -> usize {
        self.unprotected.num_bits() + self.protected.num_bits()
    }

    fn set_1(&mut self, bit_index: usize) {
        let unprotected_count = self.unprotected.num_bits();
        if bit_index < unprotected_count {
            self.unprotected.set_1(bit_index);
        } else {
            self.protected.set_1(bit_index - unprotected_count);
        }
    }

    fn set_0(&mut self, bit_index: usize) {
        let unprotected_count = self.unprotected.num_bits();
        if bit_index < unprotected_count {
            self.unprotected.set_0(bit_index);
        } else {
            self.protected.set_1(bit_index - unprotected_count);
        }
    }

    fn is_1(&self, bit_index: usize) -> bool {
        let unprotected_count = self.unprotected.num_bits();
        if bit_index < unprotected_count {
            self.unprotected.is_1(bit_index)
        } else {
            self.protected.is_1(bit_index - unprotected_count)
        }
    }

    fn flip_bit(&mut self, bit_index: usize) {
        let unprotected_count = self.unprotected.num_bits();
        if bit_index < unprotected_count {
            self.unprotected.flip_bit(bit_index)
        } else {
            self.protected.flip_bit(bit_index - unprotected_count)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitPatternEncoding {
    pub(crate) data: BitPatternEncodingData,
    pub(crate) bits_per_chunk: usize,
    pub(crate) pattern: BitPattern,
}

impl BitPatternEncoding {
    pub fn decode_into<B>(self, buffer: &mut B) -> Vec<bool>
    where
        B: BitBuffer,
    {
        let BitPatternEncoding {
            data:
                BitPatternEncodingData {
                    unprotected,
                    protected,
                },
            bits_per_chunk,
            pattern,
        } = self;

        let params = pattern
            .buffer_params(buffer)
            .expect("the pattern must be valid if it was used successfully for encoding");

        assert_eq!(params.num_unprotected_bits(), unprotected.num_bits());

        let (protected, ded_results) = protected.decode_chunks(bits_per_chunk);

        {
            let expected_num_chunks = chunks::num_chunks(params.num_protected_bits, bits_per_chunk);
            // these values are the values from before encoding.
            let expected_num_protected_bits_in_chunks = expected_num_chunks * bits_per_chunk;
            assert_eq!(expected_num_protected_bits_in_chunks, protected.num_bits());
        }

        let mut protected_i = 0;
        let mut unprotected_i = 0;
        for buffer_i in 0..buffer.num_bits() {
            let mask_i = buffer_i % pattern.length;

            if pattern.mask.contains(&mask_i) {
                if protected.is_1(protected_i) {
                    buffer.set_1(buffer_i);
                } else {
                    buffer.set_0(buffer_i);
                }
                protected_i += 1;
            } else {
                if unprotected.is_1(unprotected_i) {
                    buffer.set_1(buffer_i);
                } else {
                    buffer.set_0(buffer_i);
                }
                unprotected_i += 1;
            }
        }

        ded_results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creation() {
        let pattern = BitPattern::new([1, 3, 3, 4], 6).unwrap();
        assert_eq!(pattern.length, 6);
        assert_eq!(pattern.mask.len(), 3);

        assert_eq!(
            BitPattern::new([1, 2, 6], 6),
            Err(CreationError::InvalidLength {
                // 7 because 6 is the largest index and indices start from 0.
                expected_at_least: 7,
            })
        );

        BitPattern::new([1, 2, 6], 7).unwrap();

        assert_eq!(BitPattern::new([1, 2, 6], 0), Err(CreationError::Empty));
        assert_eq!(BitPattern::new([], 9), Err(CreationError::Empty));

        assert_eq!(BitPattern::new([0], 1), Err(CreationError::AllProtected));
        assert_eq!(BitPattern::new([0, 1], 2), Err(CreationError::AllProtected));
    }

    #[test]
    fn partition() {
        let data = 0b1111010100001010u16;
        let pattern = BitPattern::new([1, 3], 8).unwrap();

        let (unprotected, protected) = pattern.partition(&data, 4).unwrap();

        // Two bits repeated twice.
        assert_eq!(protected.num_bits(), 4);
        // chunk size 4, 4 bits total.
        assert_eq!(protected.num_chunks(), 1);

        assert_eq!(unprotected.num_bits(), 12);

        assert!(protected.is_1(0));
        assert!(protected.is_1(1));
        assert!(protected.is_0(2));
        assert!(protected.is_0(3));

        for i in 0..6 {
            assert!(unprotected.is_0(i))
        }

        for i in 6..12 {
            assert!(unprotected.is_1(i))
        }
    }

    #[test]
    fn invalid_pattern() {
        let data = 0u16;
        let pattern = BitPattern::new([0], 7).unwrap();

        assert_eq!(pattern.partition(&data, 2), Err(LengthMismatch));
    }

    #[test]
    fn repeating_pattern() {
        let data = 0u16;

        let pattern_a = BitPattern::new([0, 8], 16).unwrap();
        let pattern_b = BitPattern::new([0], 8).unwrap();

        assert_eq!(pattern_a.partition(&data, 2), pattern_b.partition(&data, 2));
    }

    #[test]
    fn chunks_with_padding() {
        let data = 0b0000001100000011u16;

        let pattern = BitPattern::new([0, 1], 8).unwrap();

        let (non_protected, protected) = pattern.partition(&data, 3).unwrap();

        // - 4 bits protected in total
        // - 3 bits per chunk
        // - 2 chunks to fit the data
        // - 6 bits in total
        assert_eq!(protected.num_chunks(), 2);
        assert_eq!(protected.num_bits(), 6);

        for i in 0..4 {
            assert!(protected.is_1(i))
        }

        for i in 4..6 {
            assert!(protected.is_0(i))
        }

        assert!(non_protected.bits().all(|is_1| !is_1));
    }

    #[test]
    fn encode_decode() {
        fn test(chunk_size: usize) {
            let data = [0u32; 16];

            let pattern = BitPattern::new([0], 8).unwrap();

            let encoded = pattern.encode(&data, chunk_size).unwrap();

            let mut decoded = [u32::MAX; 16];

            assert_eq!(
                pattern.buffer_params(&data),
                pattern.buffer_params(&decoded)
            );

            encoded.decode_into(&mut decoded);

            assert_eq!(data, decoded);
        }

        for i in 1..20 {
            test(i);
        }
    }
}
