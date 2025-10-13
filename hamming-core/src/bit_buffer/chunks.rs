use rayon::prelude::*;

use crate::encoding::decode_into;
use crate::wrapper::limited::bytes_to_store_n_bits;
use crate::{
    prelude::*,
    wrapper::{Limited, NonUniformSequence},
};

type ByteChunk = Vec<u8>;
type DynChunk = Limited<Vec<u8>>;

#[inline]
fn num_chunks(buffer_size: usize, chunk_size: usize) -> usize {
    buffer_size / chunk_size + if buffer_size % chunk_size == 0 { 0 } else { 1 }
}

/// A [`BitBuffer`] that's chunked into chunks that are multiples of 8 bits.
///
/// Should be prefered over [`DynChunks`] (if possible) due to performance reasons.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ByteChunks(NonUniformSequence<Vec<ByteChunk>>);

impl ByteChunks {
    /// Create chunks from the `buffer`.
    pub fn from_buffer<T>(buffer: T, bytes_per_chunk: usize) -> Self
    where
        T: ByteChunkedBitBuffer,
    {
        let input_size = buffer.num_bytes();
        let num_chunks = num_chunks(input_size, bytes_per_chunk);

        let mut output_buffer = NonUniformSequence(vec![vec![0u8; bytes_per_chunk]; num_chunks]);
        let num_copied = buffer.copy_into_chunked(0, &mut output_buffer);
        assert_eq!(num_copied, input_size);

        Self(output_buffer)
    }

    /// Encode all chunks in parallel.
    pub fn encode_chunks(&self) -> DynChunks {
        let output_buffer = self
            .0
             .0
            .par_iter()
            .map(|chunk| chunk.encode())
            .collect::<Vec<_>>();

        // FIXME: Replace NonUniformSequence with a uniform counterpart because all the buffers have
        // the same size and the overhead is pointless.
        DynChunks(NonUniformSequence(output_buffer))
    }

    /// Get the number of chunks.
    pub fn num_chunks(&self) -> usize {
        self.0 .0.len()
    }
}

/// A [`BitBuffer`] that's chunked into chunks of any size.
///
/// [`ByteChunks`] should be prefered (if possible) due to performance reasons.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynChunks(NonUniformSequence<Vec<DynChunk>>);

impl DynChunks {
    /// Create new dynamic chunks from the `buffer`.
    pub fn from_buffer<T>(buffer: T, bits_per_chunk: usize) -> Self
    where
        T: BitBuffer,
    {
        assert!(bits_per_chunk > 0);

        let input_size = buffer.num_bits();
        let bytes_per_chunk = bytes_to_store_n_bits(bits_per_chunk);
        let num_chunks = num_chunks(input_size, bits_per_chunk);

        // FIXME: Replace NonUniformSequence with a uniform counterpart because all the buffers have
        // the same size and the overhead is pointless.
        let mut output_buffer = NonUniformSequence(vec![
            vec![0u8; bytes_per_chunk]
                .into_limited(bits_per_chunk);
            num_chunks
        ]);

        let num_copied = buffer.copy_into(0, &mut output_buffer);
        assert_eq!(num_copied, input_size);

        Self(output_buffer)
    }

    /// Encode all chunks in parallel.
    pub fn encode_chunks(&self) -> DynChunks {
        let output_buffer = self
            .0
             .0
            .par_iter()
            .map(|chunk| chunk.encode())
            .collect::<Vec<_>>();

        // FIXME: Replace NonUniformSequence with a uniform counterpart because all the buffers have
        // the same size and the overhead is pointless.
        DynChunks(NonUniformSequence(output_buffer))
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
    pub fn decode_chunks_dyn(self, num_chunk_data_bits: usize) -> (DynChunks, Vec<bool>) {
        let num_bytes = num_chunk_data_bits / 8;

        let output_buffer =
            vec![Limited::new(vec![0u8; num_bytes], num_chunk_data_bits); self.num_chunks()];
        let (decoded_output, ded_results) = self
            .0
             .0
            .into_par_iter()
            .zip(output_buffer)
            .map(|(mut source, mut dest)| {
                let result = decode_into(&mut source, &mut dest);
                (dest, result)
            })
            .collect::<(Vec<_>, Vec<_>)>();

        // FIXME: Replace NonUniformSequence with a uniform counterpart because all the buffers have
        // the same size and the overhead is pointless.
        (DynChunks(NonUniformSequence(decoded_output)), ded_results)
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
    fn decode_chunks_byte(self, num_chunk_data_bits: usize) -> (ByteChunks, Vec<bool>) {
        let output_buffer = vec![vec![0u8; num_chunk_data_bits]; self.num_chunks()];
        let (decoded_output, results) = self
            .0
             .0
            .into_par_iter()
            .zip(output_buffer)
            .map(|(mut source, mut dest)| {
                let result = decode_into(&mut source, &mut dest);
                (dest, result)
            })
            .collect::<(Vec<_>, Vec<_>)>();

        (ByteChunks(NonUniformSequence(decoded_output)), results)
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
    pub fn decode_chunks(self, num_chunk_data_bits: usize) -> (Chunks, Vec<bool>) {
        if num_chunk_data_bits % 8 == 0 {
            let num_data_bytes = num_chunk_data_bits / 8;
            let (chunks, ded_results) = self.decode_chunks_byte(num_data_bytes);
            (Chunks::Byte(chunks), ded_results)
        } else {
            let (chunks, ded_results) = self.decode_chunks_dyn(num_chunk_data_bits);
            (Chunks::Dyn(chunks), ded_results)
        }
    }

    /// Get the number of chunks.
    pub fn num_chunks(&self) -> usize {
        self.0 .0.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Chunks {
    Byte(ByteChunks),
    Dyn(DynChunks),
}

impl Chunks {
    /// Encode all chunks in parallel.
    pub fn encode_chunks(&self) -> DynChunks {
        match self {
            Chunks::Byte(byte_chunks) => byte_chunks.encode_chunks(),
            Chunks::Dyn(dyn_chunks) => dyn_chunks.encode_chunks(),
        }
    }

    /// Create chunks from the `buffer`.
    pub fn from_buffer<T>(buffer: T, bits_per_chunk: usize) -> Self
    where
        T: ByteChunkedBitBuffer,
    {
        if bits_per_chunk % 8 == 0 {
            Chunks::Byte(buffer.to_byte_chunks(bits_per_chunk / 8))
        } else {
            Chunks::Dyn(buffer.to_dyn_chunks(bits_per_chunk))
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
    use super::*;

    #[test]
    fn bytes_per_bits() {
        assert_eq!(bytes_to_store_n_bits(0), 0);
        assert_eq!(bytes_to_store_n_bits(1), 1);
        assert_eq!(bytes_to_store_n_bits(8), 1);
        assert_eq!(bytes_to_store_n_bits(9), 2);
        assert_eq!(bytes_to_store_n_bits(16), 2);
        assert_eq!(bytes_to_store_n_bits(17), 3);
        assert_eq!(bytes_to_store_n_bits(24), 3);
        assert_eq!(bytes_to_store_n_bits(25), 4);
    }

    #[test]
    fn dyn_chunks_creation_and_restore() {
        let source = [0xffffu16; 3];
        let chunks = source.to_dyn_chunks(7);
        // The original is 6 bytes -> 48 bits long;
        // The chunk size is 7 bits.
        // We need 7 chunks -> 49 (7*7) bits to store the original data.
        assert_eq!(chunks.num_chunks(), 7);
        assert_eq!(chunks.num_bits(), 49);

        let mut bytes = chunks.0 .0.clone().into_iter();

        for _ in 0..6 {
            assert_eq!(bytes.next().unwrap().into_inner(), vec![0b01111111])
        }

        // The last byte should store 48 - 7 * 6 = 6 bits
        assert_eq!(bytes.next().unwrap().into_inner(), vec![0b00111111]);
        assert_eq!(bytes.next(), None);

        let mut restored = [0u16; 3];
        let copied = chunks.copy_into(0, &mut restored);
        assert_eq!(copied, restored.num_bits());

        let chunks = source.to_dyn_chunks(9);
        // The original is 6 bytes -> 48 bits long;
        // The chunk size is 9 bits.
        // We need 6 bytes -> 54 (9*6) bits to store the original data.
        assert_eq!(chunks.num_chunks(), 6);
        assert_eq!(chunks.num_bits(), 54);

        let mut bytes = chunks.0 .0.clone().into_iter();

        for _ in 0..5 {
            assert_eq!(bytes.next().unwrap().into_inner(), vec![0xff, 0b00000001])
        }

        // The last byte should store 48 - 9 * 5 = 3 bits
        assert_eq!(bytes.next().unwrap().into_inner(), vec![0b00000111, 0]);
        assert_eq!(bytes.next(), None);

        let mut restored = [0u16; 3];
        let copied = chunks.copy_into(0, &mut restored);
        assert_eq!(copied, restored.num_bits());
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

        let mut bytes = chunks.0 .0.clone().into_iter();

        for _ in 0..6 {
            assert_eq!(bytes.next().unwrap(), vec![0xff])
        }
        assert_eq!(bytes.next(), None);

        let mut restored = [0u16; 3];
        let copied = chunks.copy_into(0, &mut restored);
        assert_eq!(copied, restored.num_bits());

        let chunks = source.to_byte_chunks(2);
        // The original is 6 bytes -> 48 bits long;
        // The chunk size is 2 bytes.
        // 3 chunks are used to store the data.
        assert_eq!(chunks.num_chunks(), 3);
        assert_eq!(chunks.num_bits(), 48);

        let mut bytes = chunks.0 .0.clone().into_iter();

        for _ in 0..3 {
            assert_eq!(bytes.next().unwrap(), vec![0xff, 0xff])
        }
        assert_eq!(bytes.next(), None);

        let mut restored = [0u16; 3];
        let copied = chunks.copy_into(0, &mut restored);
        assert_eq!(copied, restored.num_bits());
    }
}
