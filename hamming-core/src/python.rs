//! Python bindings

use crate::{
    bit_buffer::{chunks::DynChunks, CopyIntoResult},
    encoding::num_encoded_bits,
    prelude::*,
};
use numpy::{PyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::buffers::{Limited, NonUniformSequence};

type OutputArr<'py, T> = Bound<'py, PyArray1<T>>;
type InputArr<'py, T> = PyReadonlyArrayDyn<'py, T>;

/// Helper for transforming numpy arrays received from python.
fn prep_input_array<'py, T>(input: InputArr<'py, T>) -> Vec<T>
where
    T: numpy::Element + Copy,
{
    input.as_array().into_iter().copied().collect::<Vec<T>>()
}

/// Helper for transforming lists of numpy arrays received from python.
fn prep_input_array_list<'py, T>(input: Vec<InputArr<'py, T>>) -> NonUniformSequence<Vec<Vec<T>>>
where
    T: numpy::Element + Copy,
{
    NonUniformSequence(
        input
            .into_iter()
            .map(prep_input_array)
            .collect::<Vec<Vec<T>>>(),
    )
}

fn array_list_fi_generic<'py, T>(
    py: Python<'py>,
    input: Vec<InputArr<T>>,
    faults_count: usize,
    input_bit_limit: Option<usize>,
) -> PyResult<Vec<OutputArr<'py, T>>>
where
    T: numpy::Element + Copy + SizedBitBuffer,
{
    let buffer = prep_input_array_list(input);
    let input_bits_count = buffer.num_bits();
    let bit_limit = input_bit_limit.unwrap_or(input_bits_count);
    let Some(mut buffer) = Limited::new(buffer, bit_limit) else {
        return Err(PyValueError::new_err(format!(
            "`input_bit_limit` ({}) cannot be larger than the number of bits in the buffer ({})",
            bit_limit, input_bits_count
        )));
    };

    let num_bits = buffer.num_bits();
    if faults_count > num_bits {
        return Err(PyValueError::new_err(format!(
            "Buffer has {} bits, cannot flip {}",
            num_bits, faults_count
        )));
    }

    buffer.flip_n_bits(faults_count);

    Ok(buffer
        .into_inner()
        .0
        .into_iter()
        .map(|arr| PyArray1::from_vec(py, arr))
        .collect::<Vec<_>>())
}

#[pyfunction]
pub fn f32_array_list_fi<'py>(
    py: Python<'py>,
    input: Vec<InputArr<'py, f32>>,
    faults_count: usize,
    input_bit_limit: Option<usize>,
) -> PyResult<Vec<OutputArr<'py, f32>>> {
    array_list_fi_generic(py, input, faults_count, input_bit_limit)
}

#[pyfunction]
pub fn u16_array_list_fi<'py>(
    py: Python<'py>,
    input: Vec<InputArr<'py, u16>>,
    faults_count: usize,
    input_bit_limit: Option<usize>,
) -> PyResult<Vec<OutputArr<'py, u16>>> {
    array_list_fi_generic(py, input, faults_count, input_bit_limit)
}

#[inline]
fn compare_f32_vecs_bitwise(a: Vec<f32>, b: Vec<f32>) -> Vec<u32> {
    a.into_par_iter()
        .zip(b)
        .map(|(a_item, b_item)| a_item.to_bits() ^ b_item.to_bits())
        .filter(|x| *x > 0)
        .collect::<Vec<u32>>()
}

#[inline]
fn compare_u16_vecs_bitwise(a: Vec<u16>, b: Vec<u16>) -> Vec<u16> {
    a.into_par_iter()
        .zip(b)
        .map(|(a_item, b_item)| a_item ^ b_item)
        .filter(|x| *x > 0)
        .collect::<Vec<u16>>()
}

/// Computes bit masks for non-matching bits for all items.
#[pyfunction]
pub fn compare_array_list_bitwise_f32<'py>(
    _py: Python<'py>,
    a: Vec<InputArr<f32>>,
    b: Vec<InputArr<f32>>,
) -> PyResult<Vec<u32>> {
    let a = a
        .into_iter()
        .flat_map(|x| x.as_array().into_iter().copied().collect::<Vec<f32>>())
        .collect::<Vec<_>>();
    let b = b
        .into_iter()
        .flat_map(|x| x.as_array().into_iter().copied().collect::<Vec<f32>>())
        .collect::<Vec<_>>();

    if a.len() != b.len() {
        return Err(PyValueError::new_err(
            "The number of items in `a` and `b` don't match",
        ));
    }

    Ok(compare_f32_vecs_bitwise(a, b))
}

/// Computes bit masks for non-matching bits for all items.
#[pyfunction]
pub fn compare_array_list_bitwise_u16<'py>(
    _py: Python<'py>,
    a: Vec<InputArr<u16>>,
    b: Vec<InputArr<u16>>,
) -> PyResult<Vec<u16>> {
    let a = a
        .into_iter()
        .flat_map(|x| x.as_array().into_iter().copied().collect::<Vec<u16>>())
        .collect::<Vec<_>>();
    let b = b
        .into_iter()
        .flat_map(|x| x.as_array().into_iter().copied().collect::<Vec<u16>>())
        .collect::<Vec<_>>();

    if a.len() != b.len() {
        return Err(PyValueError::new_err(
            "The number of items in `a` and `b` don't match",
        ));
    }

    Ok(compare_u16_vecs_bitwise(a, b))
}

fn encode_full_generic<'py, B>(
    py: Python<'py>,
    buffer: B,
    bits_per_chunk: usize,
) -> (OutputArr<'py, u8>, usize)
where
    B: ByteChunkedBitBuffer,
{
    let encoded_chunks = buffer.to_chunks(bits_per_chunk).encode_chunks();

    let encoded_bits_count = encoded_chunks.num_bits();
    let mut output_buffer = Limited::bytes(encoded_bits_count);

    let result = encoded_chunks.copy_into(&mut output_buffer);
    assert_eq!(
        result,
        CopyIntoResult::done(encoded_bits_count),
        "Copying failed"
    );

    (
        PyArray1::from_vec(py, output_buffer.into_inner()),
        encoded_bits_count,
    )
}

/// Encode a all bits of a buffer of 32 bit floats.
///
/// Returns a tuple containing:
/// - A uint8 array with the encoded bits
/// - The number of useful bits in the encoded buffer
///
/// These need to be given to the decoding function to restore the original representation.
#[pyfunction]
pub fn encode_full_f32<'py>(
    py: Python<'py>,
    input: Vec<InputArr<f32>>,
    bits_per_chunk: usize,
) -> (OutputArr<'py, u8>, usize) {
    let buffer = prep_input_array_list(input);

    encode_full_generic(py, buffer, bits_per_chunk)
}

/// Encode a all bits of a buffer of 16 bit unsigned integers.
///
/// Returns a tuple containing:
/// - A uint8 array with the encoded bits
/// - The number of useful bits in the encoded buffer
///
/// These need to be given to the decoding function to restore the original representation.
#[pyfunction]
pub fn encode_full_u16<'py>(
    py: Python<'py>,
    input: Vec<InputArr<u16>>,
    bits_per_chunk: usize,
) -> (OutputArr<'py, u8>, usize) {
    let buffer = prep_input_array_list(input);

    encode_full_generic(py, buffer, bits_per_chunk)
}

pub fn decode_full_generic<'py, I, O>(
    py: Python<'py>,
    input_buffer: I,
    mut output_buffer: NonUniformSequence<Vec<Vec<O>>>,
    encoded_bits_count: usize,
    bits_per_chunk: usize,
) -> PyResult<(Vec<OutputArr<'py, O>>, Vec<bool>)>
where
    I: BitBuffer,
    O: SizedBitBuffer + numpy::Element,
{
    let input_num_bits = input_buffer.num_bits();
    let Some(input_buffer) = Limited::new(input_buffer, encoded_bits_count) else {
        return Err(PyValueError::new_err(format!(
            "Got an `encoded_bits_count` ({}) which larger than the actual number of bits than the encoded buffer ({}).",
            encoded_bits_count,
            input_num_bits,
        )));
    };

    let bits_per_encoded_chunk = num_encoded_bits(bits_per_chunk);
    let input_chunks = DynChunks::from_buffer(&input_buffer, bits_per_encoded_chunk);

    if input_chunks.num_bits() != input_buffer.num_bits() {
        return Err(PyValueError::new_err("The number of bits in the limited input buffer and chunked input doesn't match (extra padding was added). \
This means one of the input parameters is incorrect but there's no way to tell which one."));
    }

    let (output_chunks, decoding_resuls) = input_chunks.decode_chunks(bits_per_chunk);
    let copy_result = output_chunks.copy_into(&mut output_buffer);
    assert_eq!(copy_result.bits_copied, output_buffer.num_bits(), "these must match because the chunked buffer can potentially only have more bits, not less.",);

    Ok((
        output_buffer
            .0
            .into_iter()
            .map(|vec| PyArray1::from_vec(py, vec))
            .collect(),
        decoding_resuls,
    ))
}

/// Decode a list of float32 values.
///
/// `decoded_array_element_counts` stores the number of items in each array in the input list for
/// the encoding function.
#[pyfunction]
pub fn decode_full_f32<'py>(
    py: Python<'py>,
    encoded: InputArr<u8>,
    encoded_bits_count: usize,
    bits_per_chunk: usize,
    decoded_array_element_counts: Vec<usize>,
) -> PyResult<(Vec<OutputArr<'py, f32>>, Vec<bool>)> {
    let input_buffer = prep_input_array(encoded);

    let output_buffer = NonUniformSequence(
        decoded_array_element_counts
            .iter()
            .map(|&numel| vec![0f32; numel])
            .collect::<Vec<_>>(),
    );

    decode_full_generic(
        py,
        input_buffer,
        output_buffer,
        encoded_bits_count,
        bits_per_chunk,
    )
}

#[pyfunction]
pub fn decode_full_u16<'py>(
    py: Python<'py>,
    encoded: InputArr<u8>,
    encoded_bits_count: usize,
    bits_per_chunk: usize,
    decoded_array_element_counts: Vec<usize>,
) -> PyResult<(Vec<OutputArr<'py, u16>>, Vec<bool>)> {
    let input_buffer = prep_input_array(encoded);

    let output_buffer = NonUniformSequence(
        decoded_array_element_counts
            .iter()
            .map(|&numel| vec![0u16; numel])
            .collect::<Vec<_>>(),
    );

    decode_full_generic(
        py,
        input_buffer,
        output_buffer,
        encoded_bits_count,
        bits_per_chunk,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_f32_vec() {
        let a = [0b10001101u32, 0b10000001u32]
            .into_iter()
            .map(f32::from_bits)
            .collect();
        let b = [0b10011100u32, 0b00000001u32]
            .into_iter()
            .map(f32::from_bits)
            .collect();

        let expected = vec![0b00010001u32, 0b10000000u32];

        assert_eq!(compare_f32_vecs_bitwise(a, b), expected);
    }

    #[test]
    fn compare_u16_vec() {
        let a = vec![0b10001101u16, 0b10000001u16];
        let b = vec![0b10011100u16, 0b00000001u16];

        let expected = vec![0b00010001u16, 0b10000000u16];

        assert_eq!(compare_u16_vecs_bitwise(a, b), expected);
    }
}
