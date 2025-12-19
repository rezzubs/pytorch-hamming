#![no_main]

use faultforge::{bit_buffer::chunks::Chunks, prelude::*};
use libfuzzer_sys::{arbitrary::Arbitrary, fuzz_target, Corpus};

#[derive(Debug, Arbitrary)]
struct Input {
    buf: Vec<u8>,
    chunk_size: usize,
    faults_count: usize,
}

fuzz_target!(|input: Input| -> Corpus {
    let Input {
        buf,
        chunk_size,
        faults_count,
    } = input;

    if buf.is_empty() {
        return Corpus::Reject;
    }

    if chunk_size > buf.bits_count() || chunk_size == 0 {
        return Corpus::Reject;
    }

    let chunks = buf.to_chunks(chunk_size).unwrap();

    let mut encoded_chunks = chunks.encode_chunks();

    if faults_count > encoded_chunks.bits_count() {
        return Corpus::Reject;
    }

    encoded_chunks.flip_n_bits(faults_count).unwrap();

    let (decoded, _results) = encoded_chunks.decode_chunks(chunk_size).unwrap();

    let mut output_buffer = vec![0u32; buf.len()];
    let _ = match decoded {
        Chunks::Byte(byte_chunks) => {
            byte_chunks
                .copy_into_chunked(&mut output_buffer)
                .units_copied
                * 8
        }
        Chunks::Dyn(dyn_chunks) => dyn_chunks.copy_into(&mut output_buffer).units_copied,
    };

    Corpus::Keep
});
