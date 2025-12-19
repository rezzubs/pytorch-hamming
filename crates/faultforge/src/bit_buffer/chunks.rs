use rayon::prelude::*;

use crate::buffers::uniform::NonMatchingIndex;
use crate::encoding::secded::decode_into;
use crate::{
    buffers::{Limited, UniformSequence},
    prelude::*,
};

pub type ByteChunk = Vec<u8>;
pub type DynChunk = Limited<Vec<u8>>;

/// The given configuration would result in invalid chunks.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InvalidChunks {
    #[error("Cannot chunk an empty buffer")]
    Empty,
    #[error("The chunk size has to be non-zero")]
    ZeroChunksize,
}

/// How many chunks are required to store a buffer with `chunk_size` bits per chunk.
#[inline]
pub fn chunks_count(buffer_size: usize, chunk_size: usize) -> Result<usize, InvalidChunks> {
    if buffer_size == 0 {
        return Err(InvalidChunks::Empty);
    }

    if chunk_size == 0 {
        return Err(InvalidChunks::ZeroChunksize);
    }

    Ok(buffer_size / chunk_size
        + if buffer_size.is_multiple_of(chunk_size) {
            0
        } else {
            1
        })
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecodeError {
    #[error(
        "The data bits count must be invalid because a length mismatch was detected during decoding: {0}"
    )]
    InvalidDataBitsCount(#[source] crate::encoding::secded::DecodeError),
}

/// A [`BitBuffer`] that's chunked into chunks that are multiples of 8 bits.
///
/// Should be prefered over [`DynChunks`] (if possible) due to performance reasons.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ByteChunks(UniformSequence<Vec<ByteChunk>>);

impl ByteChunks {
    /// Create new chunks with all bits initialized to zero.
    ///
    /// Returns [`None`] if `bytes_count` is 0.
    pub fn zero(bytes_count: usize, bytes_per_chunk: usize) -> Result<Self, InvalidChunks> {
        let chunks_count = chunks_count(bytes_count, bytes_per_chunk)?;

        Ok(Self(UniformSequence::new_unchecked(
            vec![vec![0u8; bytes_per_chunk]; chunks_count],
            bytes_per_chunk * 8,
            chunks_count,
        )))
    }

    /// Create chunks from the `buffer`.
    ///
    /// If the number of bytes in the buffer isn't a multiple of the number of bytes per chunk then
    /// it will result in a number of bytes of (essentially useless) padding at the end of the final
    /// chunk.
    ///
    /// Returns [`None`] for empty buffers.
    pub fn from_buffer<T>(buffer: &T, bytes_per_chunk: usize) -> Result<Self, InvalidChunks>
    where
        T: ByteChunkedBitBuffer,
    {
        let bytes_count = buffer.bytes_count();
        let mut output_buffer = Self::zero(bytes_count, bytes_per_chunk)?;

        let result = buffer.copy_into_chunked(&mut output_buffer);
        assert_eq!(result.units_copied, bytes_count);

        Ok(output_buffer)
    }

    /// Encode all chunks in parallel.
    #[must_use]
    pub fn encode_chunks(&self) -> DynChunks {
        let output_buffer = self
            .0
            .inner()
            .par_iter()
            .map(|chunk| {
                chunk.encode().unwrap_or_else(|err| match err {
                    super::EncodeError::Empty => unreachable!("Chunks can never be empty"),
                })
            })
            .collect::<Vec<_>>();

        DynChunks(UniformSequence::new(output_buffer).unwrap_or_else(|err| {
            unreachable!(
                "UniformSequence creation shouldn't fail as the chunk sizes are known to be the \
same unless they have been tampered with after creation. Got error {err}",
            );
        }))
    }

    /// Get the number of chunks.
    #[must_use]
    pub fn chunks_count(&self) -> usize {
        self.0.items_count()
    }

    /// Get the number of bytes per chunk.
    #[must_use]
    pub fn bytes_per_chunk(&self) -> usize {
        self.0
            .inner()
            .iter()
            .next()
            .map(|chunk| chunk.len())
            .unwrap_or(0)
    }
}

/// A [`BitBuffer`] that's chunked into chunks of any size.
///
/// [`ByteChunks`] should be prefered (if possible) due to performance reasons.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynChunks(UniformSequence<Vec<DynChunk>>);

impl DynChunks {
    pub fn zero(bits_count: usize, chunk_bits_count: usize) -> Result<Self, InvalidChunks> {
        let chunks_count = chunks_count(bits_count, chunk_bits_count)?;

        Ok(Self(UniformSequence::new_unchecked(
            vec![Limited::bytes(chunk_bits_count); chunks_count],
            chunk_bits_count,
            chunks_count,
        )))
    }
    /// Create new dynamic chunks from the `buffer`.
    ///
    /// If the number of bits in the buffer isn't a multiple of the number of bits per chunk then
    /// it will result in a number of bits of (essentially useless) padding at the end of the final
    /// chunk.
    pub fn from_buffer<T>(buffer: &T, bits_per_chunk: usize) -> Result<Self, InvalidChunks>
    where
        T: BitBuffer,
    {
        let input_size = buffer.bits_count();
        let mut output_buffer = Self::zero(input_size, bits_per_chunk)?;
        let result = buffer.copy_into(&mut output_buffer);
        assert_eq!(result.units_copied, input_size);

        Ok(output_buffer)
    }

    /// Encode all chunks in parallel.
    #[must_use]
    pub fn encode_chunks(&self) -> DynChunks {
        let output_buffer = self
            .0
            .inner()
            .par_iter()
            .map(|chunk| {
                chunk.encode().unwrap_or_else(|err| match err {
                    super::EncodeError::Empty => unreachable!("Chunks can never be empty"),
                })
            })
            .collect::<Vec<_>>();

        DynChunks(UniformSequence::new(output_buffer).unwrap_or_else(|err| {
            unreachable!(
                "UniformSequence creation shouldn't fail as the chunk sizes are known to be the \
same unless they have been tampered with after creation. Got error {err}",
            );
        }))
    }

    /// Decode the chunks in parallel.
    ///
    /// The output is guaranteed to be [`DynChunks`].
    ///
    /// The second `Vec` records double error detections.
    ///
    /// See also:
    /// - [`DynChunks::decode_chunks_byte`]
    /// - [`DynChunks::decode_chunks`]
    pub fn decode_chunks_dyn(
        self,
        chunk_data_bits_count: usize,
    ) -> Result<(DynChunks, Vec<bool>), DecodeError> {
        let chunks_count = self.chunks_count();

        let output_buffer = vec![Limited::bytes(chunk_data_bits_count); chunks_count];
        let (decoded_output, ded_results) = self
            .0
            .into_inner()
            .into_par_iter()
            .zip(output_buffer)
            .map(|(mut source, mut dest)| {
                let result = decode_into(&mut source, &mut dest).map_err(|err| match err {
                    crate::encoding::secded::DecodeError::DestEmpty => {
                        unreachable!("There is always at least one chunk")
                    }
                    crate::encoding::secded::DecodeError::LengthMismatch { .. } => {
                        DecodeError::InvalidDataBitsCount(err)
                    }
                })?;

                Ok((dest, result))
            })
            .collect::<Result<(Vec<_>, Vec<_>), DecodeError>>()?;

        Ok((
            DynChunks(UniformSequence::new_unchecked(
                decoded_output,
                chunk_data_bits_count,
                chunks_count,
            )),
            ded_results,
        ))
    }

    /// Decode the chunks in parallel.
    ///
    /// The output is guaranteed to be [`ByteChunks`].
    ///
    /// The second `Vec` records double error detections.
    ///
    /// See also:
    /// - [`DynChunks::decode_chunks_dyn`]
    /// - [`DynChunks::decode_chunks`]
    fn decode_chunks_byte(
        self,
        chunk_data_bytes_count: usize,
    ) -> Result<(ByteChunks, Vec<bool>), DecodeError> {
        let chunks_count = self.chunks_count();
        let output_buffer = vec![vec![0u8; chunk_data_bytes_count]; chunks_count];
        let (decoded_output, results) = self
            .0
            .into_inner()
            .into_par_iter()
            .zip(output_buffer)
            .map(|(mut source, mut dest)| {
                let result = decode_into(&mut source, &mut dest).map_err(|err| match err {
                    crate::encoding::secded::DecodeError::DestEmpty => {
                        unreachable!("There is always at least one chunk")
                    }
                    crate::encoding::secded::DecodeError::LengthMismatch { .. } => {
                        DecodeError::InvalidDataBitsCount(err)
                    }
                })?;

                Ok((dest, result))
            })
            .collect::<Result<(Vec<_>, Vec<_>), DecodeError>>()?;

        Ok((
            ByteChunks(UniformSequence::new_unchecked(
                decoded_output,
                chunk_data_bytes_count * 8,
                chunks_count,
            )),
            results,
        ))
    }

    /// Decode all chunks in parallel.
    ///
    /// Automatically determines the appropriate output format. For manual selection see:
    /// - [`DynChunks::decode_chunks_dyn`]
    /// - [`DynChunks::decode_chunks_byte`]
    ///
    /// The second `Vec` records double error detections.
    ///
    /// While it's simple to compute the number of required parity bits to protect a number of data
    /// bits. There is no straightforward way to compute the number of data bits from the number
    /// of encoded bits. Approximations or a brute force method will need to be used. That's why
    /// `data_bits` is given again instead.
    pub fn decode_chunks(
        self,
        chunk_data_bits_count: usize,
    ) -> Result<(Chunks, Vec<bool>), DecodeError> {
        Ok(if chunk_data_bits_count.is_multiple_of(8) {
            let data_bytes_count = chunk_data_bits_count / 8;
            let (chunks, ded_results) = self.decode_chunks_byte(data_bytes_count)?;
            (Chunks::Byte(chunks), ded_results)
        } else {
            let (chunks, ded_results) = self.decode_chunks_dyn(chunk_data_bits_count)?;
            (Chunks::Dyn(chunks), ded_results)
        })
    }

    /// Get the number of chunks.
    #[must_use]
    pub fn chunks_count(&self) -> usize {
        let chunks = self.0.items_count();
        assert!(chunks > 0);
        chunks
    }

    /// Get the number of bits per chunk.
    #[must_use]
    #[doc(alias = "chunk_size")]
    pub fn bits_per_chunk(&self) -> usize {
        self.0
            .inner()
            .iter()
            .next()
            .map(|chunk| chunk.bits_count())
            .unwrap_or(0)
    }

    #[must_use]
    pub fn into_raw(self) -> Vec<DynChunk> {
        self.0.into_inner()
    }

    pub fn from_raw(raw: Vec<DynChunk>) -> Result<Self, NonMatchingIndex> {
        UniformSequence::new(raw).map(Self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Chunks {
    Byte(ByteChunks),
    Dyn(DynChunks),
}

impl Chunks {
    /// Encode all chunks in parallel.
    #[must_use]
    pub fn encode_chunks(&self) -> DynChunks {
        match self {
            Chunks::Byte(byte_chunks) => byte_chunks.encode_chunks(),
            Chunks::Dyn(dyn_chunks) => dyn_chunks.encode_chunks(),
        }
    }

    /// Create chunks from the `buffer`.
    ///
    /// If the number of bits in the buffer isn't a multiple of the number of bits per chunk then
    /// it will result in a number of bits of (essentially useless) padding at the end of the final
    /// chunk.
    pub fn from_buffer<T>(buffer: &T, bits_per_chunk: usize) -> Result<Self, InvalidChunks>
    where
        T: ByteChunkedBitBuffer,
    {
        Ok(if bits_per_chunk.is_multiple_of(8) {
            Chunks::Byte(buffer.to_byte_chunks(bits_per_chunk / 8)?)
        } else {
            Chunks::Dyn(buffer.to_dyn_chunks(bits_per_chunk)?)
        })
    }

    /// Create new chunks with all bits initialized to zero.
    ///
    /// Returns [`None`] if `bits_count` is 0.
    pub fn zero(bits_count: usize, chunk_bits_count: usize) -> Result<Self, InvalidChunks> {
        Ok(
            if chunk_bits_count.is_multiple_of(8) && bits_count.is_multiple_of(8) {
                Chunks::Byte(ByteChunks::zero(bits_count / 8, chunk_bits_count / 8)?)
            } else {
                Chunks::Dyn(DynChunks::zero(bits_count, chunk_bits_count)?)
            },
        )
    }

    /// Get the number of chunks.
    #[must_use]
    pub fn chunks_count(&self) -> usize {
        match self {
            Chunks::Byte(byte_chunks) => byte_chunks.chunks_count(),
            Chunks::Dyn(dyn_chunks) => dyn_chunks.chunks_count(),
        }
    }

    #[must_use]
    #[doc(alias = "chunk_size")]
    pub fn bits_per_chunk(&self) -> usize {
        match self {
            Chunks::Byte(byte_chunks) => byte_chunks.bytes_per_chunk() * 8,
            Chunks::Dyn(dyn_chunks) => dyn_chunks.bits_per_chunk(),
        }
    }
}

impl BitBuffer for ByteChunks {
    fn bits_count(&self) -> usize {
        self.0.bits_count()
    }

    fn set_1(&mut self, bit_index: usize) {
        self.0.set_1(bit_index)
    }

    fn set_0(&mut self, bit_index: usize) {
        self.0.set_0(bit_index)
    }

    fn is_1(&self, bit_index: usize) -> bool {
        self.0.is_1(bit_index)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        self.0.flip_bit(bit_index)
    }
}

impl BitBuffer for DynChunks {
    fn bits_count(&self) -> usize {
        self.0.bits_count()
    }

    fn set_1(&mut self, bit_index: usize) {
        self.0.set_1(bit_index)
    }

    fn set_0(&mut self, bit_index: usize) {
        self.0.set_0(bit_index)
    }

    fn is_1(&self, bit_index: usize) -> bool {
        self.0.is_1(bit_index)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        self.0.flip_bit(bit_index)
    }
}

impl BitBuffer for Chunks {
    fn bits_count(&self) -> usize {
        match self {
            Chunks::Byte(chunks) => chunks.bits_count(),
            Chunks::Dyn(chunks) => chunks.bits_count(),
        }
    }

    fn set_1(&mut self, bit_index: usize) {
        match self {
            Chunks::Byte(chunks) => chunks.set_1(bit_index),
            Chunks::Dyn(chunks) => chunks.set_1(bit_index),
        }
    }

    fn set_0(&mut self, bit_index: usize) {
        match self {
            Chunks::Byte(chunks) => chunks.set_0(bit_index),
            Chunks::Dyn(chunks) => chunks.set_0(bit_index),
        }
    }

    fn is_1(&self, bit_index: usize) -> bool {
        match self {
            Chunks::Byte(chunks) => chunks.is_1(bit_index),
            Chunks::Dyn(chunks) => chunks.is_1(bit_index),
        }
    }

    fn flip_bit(&mut self, bit_index: usize) {
        match self {
            Chunks::Byte(chunks) => chunks.flip_bit(bit_index),
            Chunks::Dyn(chunks) => chunks.flip_bit(bit_index),
        }
    }
}

impl ByteChunkedBitBuffer for ByteChunks {
    fn bytes_count(&self) -> usize {
        self.0.bytes_count()
    }

    fn get_byte(&self, n: usize) -> u8 {
        self.0.get_byte(n)
    }

    fn set_byte(&mut self, n: usize, value: u8) {
        self.0.set_byte(n, value)
    }
}

#[cfg(test)]
mod tests {
    use crate::bit_buffer::CopyIntoResult;

    use super::*;

    #[test]
    fn dyn_chunks_creation_and_restore() {
        let source = [0xffffu16; 3];
        let chunks = source.to_dyn_chunks(7).unwrap();
        // The original is 6 bytes -> 48 bits long;
        // The chunk size is 7 bits.
        // We need 7 chunks -> 49 (7*7) bits to store the original data.
        assert_eq!(chunks.chunks_count(), 7);
        assert_eq!(chunks.bits_count(), 49);

        let mut bytes = chunks.0.clone().into_inner().into_iter();

        for _ in 0..6 {
            assert_eq!(bytes.next().unwrap().into_inner(), vec![0b01111111])
        }

        // The last byte should store 48 - 7 * 6 = 6 bits
        assert_eq!(bytes.next().unwrap().into_inner(), vec![0b00111111]);
        assert_eq!(bytes.next(), None);

        let mut restored = [0u16; 3];
        let result = chunks.copy_into(&mut restored);
        // The result will be pending because the chunks had extra padding at the end.
        assert_eq!(result, CopyIntoResult::pending(restored.bits_count()));
        assert_eq!(restored, source);

        assert_eq!(
            chunks.bits().skip(result.units_copied).count(),
            chunks.bits_count() - source.bits_count()
        );
        assert!(chunks.bits().skip(result.units_copied).all(|is_1| !is_1));

        let chunks = source.to_dyn_chunks(9).unwrap();
        // The original is 6 bytes -> 48 bits long;
        // The chunk size is 9 bits.
        // We need 6 bytes -> 54 (9*6) bits to store the original data.
        assert_eq!(chunks.chunks_count(), 6);
        assert_eq!(chunks.bits_count(), 54);

        let mut bytes = chunks.0.clone().into_inner().into_iter();

        for _ in 0..5 {
            assert_eq!(bytes.next().unwrap().into_inner(), vec![0xff, 0b00000001])
        }

        // The last byte should store 48 - 9 * 5 = 3 bits
        assert_eq!(bytes.next().unwrap().into_inner(), vec![0b00000111, 0]);
        assert_eq!(bytes.next(), None);

        let mut restored = [0u16; 3];
        let result = chunks.copy_into(&mut restored);
        assert_eq!(result.units_copied, restored.bits_count());
    }

    #[test]
    fn byte_chunks_creation_and_restore() {
        let source = [0xffffu16; 3];
        let chunks = source.to_byte_chunks(1).unwrap();
        // The original is 6 bytes -> 48 bits long;
        // The chunk size is 1 byte.
        // 6 chunks are used to store the data.
        assert_eq!(chunks.chunks_count(), 6);
        assert_eq!(chunks.bits_count(), 48);

        let mut bytes = chunks.0.clone().into_inner().into_iter();

        for _ in 0..6 {
            assert_eq!(bytes.next().unwrap(), vec![0xff])
        }
        assert_eq!(bytes.next(), None);

        let mut restored = [0u16; 3];
        let result = chunks.copy_into(&mut restored);
        assert_eq!(result, CopyIntoResult::done(restored.bits_count()));

        let chunks = source.to_byte_chunks(2).unwrap();
        // The original is 6 bytes -> 48 bits long;
        // The chunk size is 2 bytes.
        // 3 chunks are used to store the data.
        assert_eq!(chunks.chunks_count(), 3);
        assert_eq!(chunks.bits_count(), 48);

        let mut bytes = chunks.0.clone().into_inner().into_iter();

        for _ in 0..3 {
            assert_eq!(bytes.next().unwrap(), vec![0xff, 0xff])
        }
        assert_eq!(bytes.next(), None);

        let mut restored = [0u16; 3];
        let result = chunks.copy_into(&mut restored);
        assert_eq!(result, CopyIntoResult::done(restored.bits_count()));
    }

    #[test]
    fn byte_chunked_encoding() {
        let source = [123.123f32, std::f32::consts::PI, 0.001, 10000.123];
        let expected_bytes_count = 4 * 4;
        assert_eq!(source.bytes_count(), expected_bytes_count);

        let chunk_size = 16;
        let chunks_count_expected = 8;
        let expected_bytes_per_chunk = chunk_size / chunks_count_expected;
        let chunks = source.to_chunks(chunk_size).unwrap();

        match chunks {
            Chunks::Byte(ref byte_chunks) => {
                assert_eq!(byte_chunks.bytes_count(), expected_bytes_count);
                assert_eq!(byte_chunks.chunks_count(), chunks_count_expected);
                assert_eq!(
                    byte_chunks.0.inner().first().map(|x| x.len()),
                    Some(expected_bytes_per_chunk)
                );
            }
            Chunks::Dyn(dyn_chunks) => panic!("Expected byte chunks, got {:?}", dyn_chunks),
        }

        let encoded = chunks.encode_chunks();
        assert_eq!(encoded.chunks_count(), chunks_count_expected);

        {
            let non_faulty = encoded.clone();
            let (raw_decoded, ded_results) = non_faulty.decode_chunks(chunk_size).unwrap();

            for success in ded_results {
                assert!(success);
            }

            let raw_decoded = match raw_decoded {
                Chunks::Byte(byte_chunks) => byte_chunks,
                Chunks::Dyn(dyn_chunks) => panic!("Expected byte chunks, got {:?}", dyn_chunks),
            };

            let mut target = [0f32; 4];
            let result = raw_decoded.copy_into_chunked(&mut target);
            assert_eq!(result.units_copied, source.bytes_count());

            assert_eq!(target, source);
        }

        for fault_index in 0..chunk_size {
            let mut faulty = encoded.clone();
            for chunk in faulty.0.inner_mut() {
                chunk.flip_bit(fault_index);
            }
            assert_ne!(encoded, faulty);

            let (raw_decoded, ded_results) = faulty.decode_chunks(chunk_size).unwrap();

            let raw_decoded = match raw_decoded {
                Chunks::Byte(byte_chunks) => byte_chunks,
                Chunks::Dyn(dyn_chunks) => panic!("Expected byte chunks, got {:?}", dyn_chunks),
            };

            if fault_index == 0 {
                assert!(ded_results.into_iter().all(|x| !x));
            } else {
                assert!(ded_results.into_iter().all(|x| x));
            }

            let mut target = [0f32; 4];
            let result = raw_decoded.copy_into_chunked(&mut target);
            assert_eq!(result.units_copied, source.bytes_count());

            assert_eq!(target, source, "failed on fault_index={fault_index}");
        }
    }

    #[test]
    fn dyn_chunked_encoding() {
        fn test_dyn(chunk_size: usize) {
            let source = [123.123f32, std::f32::consts::PI, 0.001, 10000.123];

            let chunks = source.to_chunks(chunk_size).unwrap();

            match chunks {
                Chunks::Byte(byte_chunks) => {
                    panic!("expected dyn chunks, got {:?}", byte_chunks)
                }
                Chunks::Dyn(ref dyn_chunks) => dyn_chunks
                    .0
                    .inner()
                    .iter()
                    .for_each(|chunk| assert_eq!(chunk.bits_count(), chunk_size)),
            }

            let encoded = chunks.encode_chunks();

            {
                let non_faulty = encoded.clone();
                let (raw_decoded, ded_results) = non_faulty.decode_chunks(chunk_size).unwrap();

                for success in ded_results {
                    assert!(success);
                }

                if let Chunks::Byte(byte_chunks) = raw_decoded {
                    panic!("Expected dyn chunks, got {:?}", byte_chunks)
                };

                let mut target = [0f32; 4];
                let result = raw_decoded.copy_into(&mut target);
                if source.bits_count() == chunks.bits_count() {
                    assert_eq!(result, CopyIntoResult::done(source.bits_count()));
                } else {
                    assert_eq!(result, CopyIntoResult::pending(source.bits_count()));
                }

                assert_eq!(target, source);
            }

            for fault_index in 0..chunk_size {
                let mut faulty = encoded.clone();
                for chunk in faulty.0.inner_mut() {
                    chunk.flip_bit(fault_index);
                }
                assert_ne!(encoded, faulty);

                let (raw_decoded, ded_results) = faulty.decode_chunks(chunk_size).unwrap();

                let raw_decoded = match raw_decoded {
                    Chunks::Byte(byte_chunks) => {
                        panic!("Expected dyn chunks, got {:?}", byte_chunks)
                    }
                    Chunks::Dyn(dyn_chunks) => dyn_chunks,
                };

                if fault_index == 0 {
                    assert!(ded_results.into_iter().all(|x| !x));
                } else {
                    assert!(ded_results.into_iter().all(|x| x));
                }

                let mut target = [0f32; 4];
                let result = raw_decoded.copy_into(&mut target);
                if source.bits_count() == chunks.bits_count() {
                    assert_eq!(result, CopyIntoResult::done(source.bits_count()));
                } else {
                    assert_eq!(result, CopyIntoResult::pending(source.bits_count()));
                }

                assert_eq!(target, source, "failed on fault_index={fault_index}");
            }
        }

        for i in 1usize..=129 {
            if i.is_multiple_of(2) {
                continue;
            }
            test_dyn(i);
        }
    }

    #[test]
    fn zero() {
        let zero_buffer = [0u32; 4];

        for i in 1..=16 {
            assert_eq!(
                zero_buffer.to_chunks(i).unwrap(),
                Chunks::zero(zero_buffer.bits_count(), i).unwrap()
            );
        }
    }

    #[test]
    fn invalid() {
        let buffer: UniformSequence<Vec<u8>> = UniformSequence::new(vec![]).unwrap();

        for size in 0..16 {
            assert_eq!(buffer.to_chunks(size), Err(InvalidChunks::Empty));
        }

        let buffer = 0u64;

        assert_eq!(buffer.to_chunks(0), Err(InvalidChunks::ZeroChunksize));
        assert_eq!(buffer.to_dyn_chunks(0), Err(InvalidChunks::ZeroChunksize));
        assert_eq!(buffer.to_byte_chunks(0), Err(InvalidChunks::ZeroChunksize));
    }

    #[test]
    #[doc(alias = "chunk_size")]
    fn bits_per_chunk() {
        for i in 1..=64 {
            let chunks = [0u8; 14].to_chunks(i).unwrap();
            assert_eq!(chunks.bits_per_chunk(), i);
        }
    }
}
