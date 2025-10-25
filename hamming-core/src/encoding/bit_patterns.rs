#![allow(dead_code)]

use std::collections::HashSet;

use crate::{
    bit_buffer::{
        chunks::{self, Chunks, DynChunks},
        CopyIntoResult,
    },
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
        chunk_size: usize,
    ) -> Result<(Limited<Vec<u8>>, Chunks), LengthMismatch>
    where
        B: BitBuffer,
    {
        let params = self.buffer_params(buffer)?;

        let mut protected = Chunks::zero(params.num_protected_bits, chunk_size);
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
        chunk_size: usize,
    ) -> Result<BitPatternEncoding, LengthMismatch>
    where
        B: BitBuffer,
    {
        let (unprotected, protected) = self.partition(buffer, chunk_size)?;

        Ok(BitPatternEncoding {
            protected: protected.encode_chunks(),
            unprotected,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BitPatternEncoding {
    unprotected: Limited<Vec<u8>>,
    protected: DynChunks,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
pub enum DecodeError {
    /// The length of the buffer isn't a multiple of the [`BitPattern`].
    #[error(transparent)]
    LengthMismatch(#[from] LengthMismatch),
    #[error("This buffer-pattern combo expects {expected} protected bits, got {actual}")]
    ProtectedMimatch {
        /// The number of unprotected bits in [`BitPatternEncoding`].
        actual: usize,
        /// The expected number of unprotected bits per buffer.
        expected: usize,
    },
    #[error("This buffer-pattern combo expects {expected} unprotected bits, got {actual}")]
    UnprotectedMismatch {
        /// The number of unprotected bits in [`BitPatternEncoding`].
        actual: usize,
        /// The expected number of unprotected bits per buffer.
        expected: usize,
    },
}

struct BitPatternEncodingBytes {
    /// The raw data.
    ///
    /// Protected bits come after the protected ones
    pub bytes: Limited<Vec<u8>>,
    /// Number of unprotected bits
    pub num_unprotected: usize,
    /// Number of bits in a protected chunk
    pub encoded_chunk_size: usize,
    /// Number of chunks that store the protected bits.
    pub num_encoded_chunks: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("Got a {actual_num_bits} bit buffer, but the other parameters describe a {described_num_bits} buffer")]
struct InvalidBytesDescription {
    pub actual_num_bits: usize,
    pub described_num_bits: usize,
}

impl BitPatternEncoding {
    fn decode_into<B>(
        self,
        buffer: &mut B,
        data_bits_per_chunk: usize,
        pattern: BitPattern,
    ) -> Result<Vec<bool>, DecodeError>
    where
        B: BitBuffer,
    {
        let params = pattern.buffer_params(buffer)?;

        let BitPatternEncoding {
            unprotected,
            protected,
        } = self;

        if params.num_unprotected_bits() != unprotected.num_bits() {
            return Err(DecodeError::UnprotectedMismatch {
                expected: params.num_unprotected_bits(),
                actual: unprotected.num_bits(),
            });
        }

        let (protected, ded_results) = protected.decode_chunks(data_bits_per_chunk);

        {
            let expected_num_chunks =
                chunks::num_chunks(params.num_protected_bits, data_bits_per_chunk);
            // these values are the values from before encoding.
            let expected_num_protected_bits_in_chunks = expected_num_chunks * data_bits_per_chunk;
            if expected_num_protected_bits_in_chunks != protected.num_bits() {
                return Err(DecodeError::ProtectedMimatch {
                    expected: expected_num_protected_bits_in_chunks,
                    actual: protected.num_bits(),
                });
            }
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

        Ok(ded_results)
    }

    fn to_bytes(&self) -> BitPatternEncodingBytes {
        let mut bytes = Limited::bytes(self.unprotected.num_bits() + self.protected.num_bits());

        let result = self.unprotected.copy_into(&mut bytes);
        debug_assert_eq!(result, CopyIntoResult::done(self.unprotected.num_bits()));

        let result2 = self
            .protected
            .copy_into_offset(0, result.bits_copied, &mut bytes);
        debug_assert_eq!(result2, CopyIntoResult::done(self.protected.num_bits()));
        debug_assert_eq!(result.bits_copied + result2.bits_copied, bytes.num_bits());

        BitPatternEncodingBytes {
            bytes,
            num_unprotected: self.unprotected.num_bits(),
            encoded_chunk_size: self.protected.bits_per_chunk(),
            num_encoded_chunks: self.protected.num_chunks(),
        }
    }

    fn from_bytes(bytes: &BitPatternEncodingBytes) -> Result<Self, InvalidBytesDescription> {
        let mut unprotected = Limited::bytes(bytes.num_unprotected);

        let mut protected = DynChunks::zero(
            bytes.num_encoded_chunks * bytes.encoded_chunk_size,
            bytes.encoded_chunk_size,
        );

        if bytes.bytes.num_bits() != protected.num_bits() + unprotected.num_bits() {
            return Err(InvalidBytesDescription {
                actual_num_bits: bytes.bytes.num_bits(),
                described_num_bits: protected.num_bits() + unprotected.num_bits(),
            });
        }

        let result = bytes.bytes.copy_into(&mut unprotected);
        assert_eq!(
            result,
            CopyIntoResult::pending(unprotected.num_bits()),
            concat!(
                "the buffer is confirmed to have enough space ",
                "and the result must be pending if `protected` a nonzero number of bits"
            )
        );

        let result2 = bytes
            .bytes
            .copy_into_offset(result.bits_copied, 0, &mut protected);

        assert_eq!(result2, CopyIntoResult::done(protected.num_bits()));

        Ok(Self {
            unprotected,
            protected,
        })
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

            encoded
                .decode_into(&mut decoded, chunk_size, pattern)
                .unwrap();

            assert_eq!(data, decoded);
        }

        for i in 1..20 {
            test(i);
        }
    }

    #[test]
    fn intermediate_repr_encode_decode() {
        let data: [u32; 4] = [1234, 0, 1, 454378230];

        // Protect every fourth bit
        let pattern = BitPattern::new([3], 4).unwrap();

        let chunk_size = 8;
        let encoded = pattern.encode(&data, chunk_size).unwrap();

        let ir = encoded.to_bytes();

        let encoded2 = BitPatternEncoding::from_bytes(&ir).unwrap();
        assert_eq!(encoded2, encoded);

        let mut decoded = [0u32; 4];
        encoded2
            .decode_into(&mut decoded, chunk_size, pattern)
            .unwrap();

        assert_eq!(decoded, data);
    }
}
