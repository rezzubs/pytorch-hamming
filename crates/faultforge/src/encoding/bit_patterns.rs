#![allow(dead_code)]

use std::collections::HashSet;

use crate::{
    bit_buffer::chunks::{self, Chunks, DynChunks, InvalidChunks},
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

/// Error cases for [`BitPattern::partition`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PartitionError {
    #[error(transparent)]
    LengthMisMatch(#[from] LengthMismatch),
    #[error("Failed to chunk the buffer\n-> {0}")]
    InvalidChunks(#[from] InvalidChunks),
}

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
    total_bits_count: usize,
    masks_count: usize,
    protected_bits_count: usize,
}

impl BufferParams {
    fn unprotected_bits_count(&self) -> usize {
        self.total_bits_count - self.protected_bits_count
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

        let total_bits_count = buffer.bits_count();
        if !total_bits_count.is_multiple_of(self.length) {
            return Err(LengthMismatch);
        }

        let masks_count = buffer.bits_count() / self.length;
        // NOTE: the length of the mask corresponds to the number of bits to be encoded per pattern.
        let protected_bits_count = self.mask.len() * masks_count;
        assert!(protected_bits_count <= total_bits_count);

        Ok(BufferParams {
            total_bits_count,
            masks_count,
            protected_bits_count,
        })
    }

    pub fn partition<B>(
        &self,
        buffer: &B,
        bits_per_chunk: usize,
    ) -> Result<(Limited<Vec<u8>>, Chunks), PartitionError>
    where
        B: BitBuffer,
    {
        let params = self.buffer_params(buffer)?;

        let mut protected = Chunks::zero(params.protected_bits_count, bits_per_chunk)?;
        let mut unprotected = Limited::bytes(params.unprotected_bits_count());

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
    ) -> Result<BitPatternEncoding, PartitionError>
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
    // FIXME: Consider restricting access because mutation can cause invariants
    // to break.
    pub unprotected: Limited<Vec<u8>>,
    pub protected: DynChunks,
}

impl BitBuffer for BitPatternEncodingData {
    fn bits_count(&self) -> usize {
        self.unprotected.bits_count() + self.protected.bits_count()
    }

    fn set_1(&mut self, bit_index: usize) {
        let unprotected_count = self.unprotected.bits_count();
        if bit_index < unprotected_count {
            self.unprotected.set_1(bit_index);
        } else {
            self.protected.set_1(bit_index - unprotected_count);
        }
    }

    fn set_0(&mut self, bit_index: usize) {
        let unprotected_count = self.unprotected.bits_count();
        if bit_index < unprotected_count {
            self.unprotected.set_0(bit_index);
        } else {
            self.protected.set_1(bit_index - unprotected_count);
        }
    }

    fn is_1(&self, bit_index: usize) -> bool {
        let unprotected_count = self.unprotected.bits_count();
        if bit_index < unprotected_count {
            self.unprotected.is_1(bit_index)
        } else {
            self.protected.is_1(bit_index - unprotected_count)
        }
    }

    fn flip_bit(&mut self, bit_index: usize) {
        let unprotected_count = self.unprotected.bits_count();
        if bit_index < unprotected_count {
            self.unprotected.flip_bit(bit_index)
        } else {
            self.protected.flip_bit(bit_index - unprotected_count)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitPatternEncoding {
    // TODO: Consider restricting access because mutation can cause invariants
    // to break.
    pub data: BitPatternEncodingData,
    pub pattern: BitPattern,
    pub bits_per_chunk: usize,
}

impl BitPatternEncoding {
    #[must_use]
    pub fn bits_per_chunk(&self) -> usize {
        self.bits_per_chunk
    }

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

        assert_eq!(params.unprotected_bits_count(), unprotected.bits_count());

        let (protected, ded_results) =
            protected
                .decode_chunks(bits_per_chunk)
                .unwrap_or_else(|err| match err {
                    chunks::DecodeError::InvalidDataBitsCount(_) => unreachable!(
                        "bits_per_chunk cannot be incorrect because the encoding was successful and it cannot be altered after"
                    ),
                });

        {
            let chunks_count_expected = chunks::chunks_count(params.protected_bits_count, bits_per_chunk).expect("It should be possible to decode using the same parameters that were used to encode.");
            // these values are the values from before encoding.
            let chunk_protected_bits_count_expected = chunks_count_expected * bits_per_chunk;
            assert_eq!(chunk_protected_bits_count_expected, protected.bits_count());
        }

        let mut protected_i = 0;
        let mut unprotected_i = 0;
        for buffer_i in 0..buffer.bits_count() {
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
        assert_eq!(protected.bits_count(), 4);
        // chunk size 4, 4 bits total.
        assert_eq!(protected.chunks_count(), 1);

        assert_eq!(unprotected.bits_count(), 12);

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

        assert_eq!(
            pattern.partition(&data, 2),
            Err(PartitionError::LengthMisMatch(LengthMismatch))
        );
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
        assert_eq!(protected.chunks_count(), 2);
        assert_eq!(protected.bits_count(), 6);

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
