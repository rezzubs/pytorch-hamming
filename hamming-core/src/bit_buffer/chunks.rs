use rayon::prelude::*;

use crate::encoding::decode_into;
use crate::{
    buffers::{Limited, UniformSequence},
    prelude::*,
};

type ByteChunk = Vec<u8>;
type DynChunk = Limited<Vec<u8>>;

/// How many chunks are required to store a buffer with `chunk_size` bits per chunk.
#[inline]
#[must_use]
pub fn num_chunks(buffer_size: usize, chunk_size: usize) -> usize {
    buffer_size / chunk_size
        + if buffer_size.is_multiple_of(chunk_size) {
            0
        } else {
            1
        }
}

/// A [`BitBuffer`] that's chunked into chunks that are multiples of 8 bits.
///
/// Should be prefered over [`DynChunks`] (if possible) due to performance reasons.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ByteChunks(UniformSequence<Vec<ByteChunk>>);

impl ByteChunks {
    /// Create new chunks with all bits initialized to zero.
    #[must_use]
    pub fn zero(num_bytes: usize, bytes_per_chunk: usize) -> Self {
        assert!(bytes_per_chunk > 0);

        let num_chunks = num_chunks(num_bytes, bytes_per_chunk);

        Self(UniformSequence::new_unchecked(
            vec![vec![0u8; bytes_per_chunk]; num_chunks],
            bytes_per_chunk * 8,
            num_chunks,
        ))
    }

    /// Create chunks from the `buffer`.
    ///
    /// If the number of bytes in the buffer isn't a multiple of the number of bytes per chunk then
    /// it will result in a number of bytes of (essentially useless) padding at the end of the final
    /// chunk.
    pub fn from_buffer<T>(buffer: &T, bytes_per_chunk: usize) -> Self
    where
        T: ByteChunkedBitBuffer,
    {
        let num_bytes = buffer.num_bytes();
        let mut output_buffer = Self::zero(num_bytes, bytes_per_chunk);

        let num_copied = buffer.copy_into_chunked(0, &mut output_buffer);
        assert_eq!(num_copied, num_bytes);

        output_buffer
    }

    /// Encode all chunks in parallel.
    #[must_use]
    pub fn encode_chunks(&self) -> DynChunks {
        let output_buffer = self
            .0
            .inner()
            .par_iter()
            .map(|chunk| chunk.encode())
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
    pub fn num_chunks(&self) -> usize {
        self.0.num_items()
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
    #[must_use]
    pub fn zero(num_bits: usize, bits_per_chunk: usize) -> Self {
        assert!(bits_per_chunk > 0);

        let num_chunks = num_chunks(num_bits, bits_per_chunk);

        Self(UniformSequence::new_unchecked(
            vec![Limited::bytes(bits_per_chunk); num_chunks],
            bits_per_chunk,
            num_chunks,
        ))
    }
    /// Create new dynamic chunks from the `buffer`.
    ///
    /// If the number of bits in the buffer isn't a multiple of the number of bits per chunk then
    /// it will result in a number of bits of (essentially useless) padding at the end of the final
    /// chunk.
    pub fn from_buffer<T>(buffer: &T, bits_per_chunk: usize) -> Self
    where
        T: BitBuffer,
    {
        let input_size = buffer.num_bits();
        let mut output_buffer = Self::zero(input_size, bits_per_chunk);
        let result = buffer.copy_into(&mut output_buffer);
        assert_eq!(result.bits_copied, input_size);

        output_buffer
    }

    /// Encode all chunks in parallel.
    #[must_use]
    pub fn encode_chunks(&self) -> DynChunks {
        let output_buffer = self
            .0
            .inner()
            .par_iter()
            .map(|chunk| chunk.encode())
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
    #[must_use]
    pub fn decode_chunks_dyn(self, num_chunk_data_bits: usize) -> (DynChunks, Vec<bool>) {
        let num_chunks = self.num_chunks();
        let output_buffer = vec![Limited::bytes(num_chunk_data_bits); num_chunks];
        let (decoded_output, ded_results) = self
            .0
            .into_inner()
            .into_par_iter()
            .zip(output_buffer)
            .map(|(mut source, mut dest)| {
                let result = decode_into(&mut source, &mut dest);
                (dest, result)
            })
            .collect::<(Vec<_>, Vec<_>)>();

        (
            DynChunks(UniformSequence::new_unchecked(
                decoded_output,
                num_chunk_data_bits,
                num_chunks,
            )),
            ded_results,
        )
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
    fn decode_chunks_byte(self, num_chunk_data_bytes: usize) -> (ByteChunks, Vec<bool>) {
        let num_chunks = self.num_chunks();
        let output_buffer = vec![vec![0u8; num_chunk_data_bytes]; num_chunks];
        let (decoded_output, results) = self
            .0
            .into_inner()
            .into_par_iter()
            .zip(output_buffer)
            .map(|(mut source, mut dest)| {
                let result = decode_into(&mut source, &mut dest);
                (dest, result)
            })
            .collect::<(Vec<_>, Vec<_>)>();

        (
            ByteChunks(UniformSequence::new_unchecked(
                decoded_output,
                num_chunk_data_bytes * 8,
                num_chunks,
            )),
            results,
        )
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
    #[must_use]
    pub fn decode_chunks(self, num_chunk_data_bits: usize) -> (Chunks, Vec<bool>) {
        if num_chunk_data_bits.is_multiple_of(8) {
            let num_data_bytes = num_chunk_data_bits / 8;
            let (chunks, ded_results) = self.decode_chunks_byte(num_data_bytes);
            (Chunks::Byte(chunks), ded_results)
        } else {
            let (chunks, ded_results) = self.decode_chunks_dyn(num_chunk_data_bits);
            (Chunks::Dyn(chunks), ded_results)
        }
    }

    /// Get the number of chunks.
    #[must_use]
    pub fn num_chunks(&self) -> usize {
        self.0.num_items()
    }

    /// Get the number of bits per chunk.
    #[must_use]
    pub fn bits_per_chunk(&self) -> usize {
        self.0
            .inner()
            .iter()
            .next()
            .map(|chunk| chunk.num_bits())
            .unwrap_or(0)
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
    pub fn from_buffer<T>(buffer: &T, bits_per_chunk: usize) -> Self
    where
        T: ByteChunkedBitBuffer,
    {
        if bits_per_chunk.is_multiple_of(8) {
            Chunks::Byte(buffer.to_byte_chunks(bits_per_chunk / 8))
        } else {
            Chunks::Dyn(buffer.to_dyn_chunks(bits_per_chunk))
        }
    }

    /// Create new chunks with all bits initialized to zero.
    #[must_use]
    pub fn zero(num_bits: usize, bits_per_chunk: usize) -> Self {
        if bits_per_chunk.is_multiple_of(8) && num_bits.is_multiple_of(8) {
            Chunks::Byte(ByteChunks::zero(num_bits / 8, bits_per_chunk / 8))
        } else {
            Chunks::Dyn(DynChunks::zero(num_bits, bits_per_chunk))
        }
    }

    /// Get the number of chunks.
    #[must_use]
    pub fn num_chunks(&self) -> usize {
        match self {
            Chunks::Byte(byte_chunks) => byte_chunks.num_chunks(),
            Chunks::Dyn(dyn_chunks) => dyn_chunks.num_chunks(),
        }
    }

    #[must_use]
    pub fn bits_per_chunk(&self) -> usize {
        match self {
            Chunks::Byte(byte_chunks) => byte_chunks.bytes_per_chunk() * 8,
            Chunks::Dyn(dyn_chunks) => dyn_chunks.bits_per_chunk(),
        }
    }
}

impl BitBuffer for ByteChunks {
    fn num_bits(&self) -> usize {
        self.0.num_bits()
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
    fn num_bits(&self) -> usize {
        self.0.num_bits()
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
    fn num_bits(&self) -> usize {
        match self {
            Chunks::Byte(chunks) => chunks.num_bits(),
            Chunks::Dyn(chunks) => chunks.num_bits(),
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
    fn num_bytes(&self) -> usize {
        self.0.num_bytes()
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
        let chunks = source.to_dyn_chunks(7);
        // The original is 6 bytes -> 48 bits long;
        // The chunk size is 7 bits.
        // We need 7 chunks -> 49 (7*7) bits to store the original data.
        assert_eq!(chunks.num_chunks(), 7);
        assert_eq!(chunks.num_bits(), 49);

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
        assert_eq!(result, CopyIntoResult::pending(restored.num_bits()));
        assert_eq!(restored, source);

        assert_eq!(
            chunks.bits().skip(result.bits_copied).count(),
            chunks.num_bits() - source.num_bits()
        );
        assert!(chunks.bits().skip(result.bits_copied).all(|is_1| !is_1));

        let chunks = source.to_dyn_chunks(9);
        // The original is 6 bytes -> 48 bits long;
        // The chunk size is 9 bits.
        // We need 6 bytes -> 54 (9*6) bits to store the original data.
        assert_eq!(chunks.num_chunks(), 6);
        assert_eq!(chunks.num_bits(), 54);

        let mut bytes = chunks.0.clone().into_inner().into_iter();

        for _ in 0..5 {
            assert_eq!(bytes.next().unwrap().into_inner(), vec![0xff, 0b00000001])
        }

        // The last byte should store 48 - 9 * 5 = 3 bits
        assert_eq!(bytes.next().unwrap().into_inner(), vec![0b00000111, 0]);
        assert_eq!(bytes.next(), None);

        let mut restored = [0u16; 3];
        let result = chunks.copy_into(&mut restored);
        assert_eq!(result.bits_copied, restored.num_bits());
    }

    #[test]
    fn byte_chunks_creation_and_restore() {
        let source = [0xffffu16; 3];
        let chunks = source.to_byte_chunks(1);
        // The original is 6 bytes -> 48 bits long;
        // The chunk size is 1 byte.
        // 6 chunks are used to store the data.
        assert_eq!(chunks.num_chunks(), 6);
        assert_eq!(chunks.num_bits(), 48);

        let mut bytes = chunks.0.clone().into_inner().into_iter();

        for _ in 0..6 {
            assert_eq!(bytes.next().unwrap(), vec![0xff])
        }
        assert_eq!(bytes.next(), None);

        let mut restored = [0u16; 3];
        let result = chunks.copy_into(&mut restored);
        assert_eq!(result, CopyIntoResult::done(restored.num_bits()));

        let chunks = source.to_byte_chunks(2);
        // The original is 6 bytes -> 48 bits long;
        // The chunk size is 2 bytes.
        // 3 chunks are used to store the data.
        assert_eq!(chunks.num_chunks(), 3);
        assert_eq!(chunks.num_bits(), 48);

        let mut bytes = chunks.0.clone().into_inner().into_iter();

        for _ in 0..3 {
            assert_eq!(bytes.next().unwrap(), vec![0xff, 0xff])
        }
        assert_eq!(bytes.next(), None);

        let mut restored = [0u16; 3];
        let result = chunks.copy_into(&mut restored);
        assert_eq!(result, CopyIntoResult::done(restored.num_bits()));
    }

    #[test]
    fn byte_chunked_encoding() {
        let source = [123.123f32, std::f32::consts::PI, 0.001, 10000.123];
        let expected_num_bytes = 4 * 4;
        assert_eq!(source.num_bytes(), expected_num_bytes);

        let chunk_size = 16;
        let expected_num_chunks = 8;
        let expected_bytes_per_chunk = chunk_size / expected_num_chunks;
        let chunks = source.to_chunks(chunk_size);

        match chunks {
            Chunks::Byte(ref byte_chunks) => {
                assert_eq!(byte_chunks.num_bytes(), expected_num_bytes);
                assert_eq!(byte_chunks.num_chunks(), expected_num_chunks);
                assert_eq!(
                    byte_chunks.0.inner().first().map(|x| x.len()),
                    Some(expected_bytes_per_chunk)
                );
            }
            Chunks::Dyn(dyn_chunks) => panic!("Expected byte chunks, got {:?}", dyn_chunks),
        }

        let encoded = chunks.encode_chunks();
        assert_eq!(encoded.num_chunks(), expected_num_chunks);

        {
            let non_faulty = encoded.clone();
            let (raw_decoded, ded_results) = non_faulty.decode_chunks(chunk_size);

            for success in ded_results {
                assert!(success);
            }

            let raw_decoded = match raw_decoded {
                Chunks::Byte(byte_chunks) => byte_chunks,
                Chunks::Dyn(dyn_chunks) => panic!("Expected byte chunks, got {:?}", dyn_chunks),
            };

            let mut target = [0f32; 4];
            let copied = raw_decoded.copy_into_chunked(0, &mut target);
            assert_eq!(copied, source.num_bytes());

            assert_eq!(target, source);
        }

        for fault_index in 0..chunk_size {
            let mut faulty = encoded.clone();
            for chunk in faulty.0.inner_mut() {
                chunk.flip_bit(fault_index);
            }
            assert_ne!(encoded, faulty);

            let (raw_decoded, ded_results) = faulty.decode_chunks(chunk_size);

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
            let copied = raw_decoded.copy_into_chunked(0, &mut target);
            assert_eq!(copied, source.num_bytes());

            assert_eq!(target, source, "failed on fault_index={fault_index}");
        }
    }

    #[test]
    fn dyn_chunked_encoding() {
        fn test_dyn(chunk_size: usize) {
            let source = [123.123f32, std::f32::consts::PI, 0.001, 10000.123];

            let chunks = source.to_chunks(chunk_size);

            match chunks {
                Chunks::Byte(byte_chunks) => {
                    panic!("expected dyn chunks, got {:?}", byte_chunks)
                }
                Chunks::Dyn(ref dyn_chunks) => dyn_chunks
                    .0
                    .inner()
                    .iter()
                    .for_each(|chunk| assert_eq!(chunk.num_bits(), chunk_size)),
            }

            let encoded = chunks.encode_chunks();

            {
                let non_faulty = encoded.clone();
                let (raw_decoded, ded_results) = non_faulty.decode_chunks(chunk_size);

                for success in ded_results {
                    assert!(success);
                }

                if let Chunks::Byte(byte_chunks) = raw_decoded {
                    panic!("Expected dyn chunks, got {:?}", byte_chunks)
                };

                let mut target = [0f32; 4];
                let result = raw_decoded.copy_into(&mut target);
                if source.num_bits() == chunks.num_bits() {
                    assert_eq!(result, CopyIntoResult::done(source.num_bits()));
                } else {
                    assert_eq!(result, CopyIntoResult::pending(source.num_bits()));
                }

                assert_eq!(target, source);
            }

            for fault_index in 0..chunk_size {
                let mut faulty = encoded.clone();
                for chunk in faulty.0.inner_mut() {
                    chunk.flip_bit(fault_index);
                }
                assert_ne!(encoded, faulty);

                let (raw_decoded, ded_results) = faulty.decode_chunks(chunk_size);

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
                if source.num_bits() == chunks.num_bits() {
                    assert_eq!(result, CopyIntoResult::done(source.num_bits()));
                } else {
                    assert_eq!(result, CopyIntoResult::pending(source.num_bits()));
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
    fn empty() {
        let buf: Vec<u8> = vec![];

        let chunks = buf.clone().to_chunks(7);

        assert_eq!(chunks.num_bits(), buf.num_bits());
    }

    #[test]
    fn zero() {
        let zero_buffer = [0u32; 4];

        for i in 1..=16 {
            assert_eq!(
                zero_buffer.to_chunks(i),
                Chunks::zero(zero_buffer.num_bits(), i)
            );
        }
    }

    #[test]
    fn bits_per_chunk() {
        for i in 1..=64 {
            let chunks = [0u8; 14].to_chunks(i);
            assert_eq!(chunks.bits_per_chunk(), i);
        }
    }
}
