use crate::{
    bit_buffer::chunks::DynChunks,
    buffers::{Limited, NonUniformSequence},
    encoding::{
        bit_patterns::{self, BitPattern, BitPatternEncoding},
        num_encoded_bits,
    },
    prelude::*,
    python::common::*,
};
use numpy::PyArray1;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyList},
};
use std::collections::HashSet;

#[pyclass(name = "BitPatternEncoding")]
pub struct PyBitPatternEncoding {
    unprotected: Py<PyBytes>,
    unprotected_bits_count: usize,
    // Storing PyBytes elements
    protected: Py<PyList>,
}

fn py_bit_pattern(
    protected_bits: HashSet<usize>,
    bit_pattern_length: usize,
) -> PyResult<BitPattern> {
    BitPattern::new(protected_bits.clone(), bit_pattern_length).map_err(|err| {
        PyValueError::new_err(match err {
            bit_patterns::CreationError::InvalidLength { expected_at_least } => {
                format!(
                    "Invalid length ({}) for bit pattern ({:?}), expected at least {}",
                    bit_pattern_length, protected_bits, expected_at_least
                )
            }
            bit_patterns::CreationError::AllProtected => format!(
                "The pattern {:?} with length {} covers all bits, use some other technique",
                protected_bits, bit_pattern_length,
            ),
            bit_patterns::CreationError::Empty => bit_patterns::CreationError::Empty.to_string(),
        })
    })
}

fn encode_bit_pattern_generic<'py, T>(
    py: Python<'py>,
    buffer: NonUniformSequence<Vec<Vec<T>>>,
    bit_pattern_bits: HashSet<usize>,
    bit_pattern_length: usize,
    bits_per_chunk: usize,
) -> PyResult<PyBitPatternEncoding>
where
    T: SizedBitBuffer,
{
    let pattern = py_bit_pattern(bit_pattern_bits, bit_pattern_length)?;

    let BitPatternEncoding {
        unprotected,
        protected,
    } = pattern
        .encode(&buffer, bits_per_chunk)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let unprotected_bits_count = unprotected.num_bits();

    let protected = protected
        .into_raw()
        .into_iter()
        .map(|chunk| PyBytes::new(py, &chunk.into_inner()).unbind());

    let protected = PyList::new(py, protected)?.unbind();

    Ok(PyBitPatternEncoding {
        unprotected: PyBytes::new(py, &unprotected.into_inner()).unbind(),
        unprotected_bits_count,
        protected,
    })
}

#[pyfunction]
pub fn encode_bit_pattern_f32<'py>(
    py: Python<'py>,
    input: Vec<InputArr<f32>>,
    bit_pattern_bits: HashSet<usize>,
    bit_pattern_length: usize,
    bits_per_chunk: usize,
) -> PyResult<PyBitPatternEncoding> {
    let buffer = prep_input_array_list(input);

    encode_bit_pattern_generic(
        py,
        buffer,
        bit_pattern_bits,
        bit_pattern_length,
        bits_per_chunk,
    )
}

#[pyfunction]
pub fn encode_bit_pattern_u16<'py>(
    py: Python<'py>,
    input: Vec<InputArr<u16>>,
    bit_pattern_bits: HashSet<usize>,
    bit_pattern_length: usize,
    bits_per_chunk: usize,
) -> PyResult<PyBitPatternEncoding> {
    let buffer = prep_input_array_list(input);

    encode_bit_pattern_generic(
        py,
        buffer,
        bit_pattern_bits,
        bit_pattern_length,
        bits_per_chunk,
    )
}

pub fn decode_bit_pattern_generic<'py, T>(
    py: Python<'py>,
    encoding: PyRef<PyBitPatternEncoding>,
    bit_pattern_bits: HashSet<usize>,
    bit_pattern_length: usize,
    bits_per_chunk: usize,
    mut output_buffer: NonUniformSequence<Vec<Vec<T>>>,
) -> PyResult<(Vec<OutputArr<'py, T>>, Vec<bool>)>
where
    T: numpy::Element + BitBuffer + SizedBitBuffer,
{
    let pattern = py_bit_pattern(bit_pattern_bits, bit_pattern_length)?;

    let unprotected = Vec::from(encoding.unprotected.as_bytes(py));
    let unprotected_bits_count_actual = unprotected.num_bits();
    let unprotected = Limited::new(unprotected, encoding.unprotected_bits_count).ok_or_else(|| {

        PyValueError::new_err(format!("`unprotected_bits_count` ({}) cannot be larger than the number bits in of `unprotected` ({})", encoding.unprotected_bits_count, unprotected_bits_count_actual))
    })?;

    let encoded_bits_per_chunk = num_encoded_bits(bits_per_chunk);
    let protected = encoding
        .protected
        .bind(py)
        .iter()
        .map(|chunk| -> PyResult<_> {
            let chunk = chunk.downcast_into::<PyBytes>()?;

            Limited::new(Vec::from(chunk.as_bytes()), encoded_bits_per_chunk).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "`bits_per_chunk` ({}) doesn't match the encoded chunks",
                    bits_per_chunk
                ))
            })
        })
        .collect::<PyResult<_>>()?;

    let protected =
        DynChunks::from_raw(protected).map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    let encoding = BitPatternEncoding {
        unprotected,
        protected,
    };

    let decoding_results = encoding
        .decode_into(&mut output_buffer, bits_per_chunk, pattern)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    Ok((
        output_buffer
            .0
            .into_iter()
            .map(|vec| PyArray1::from_vec(py, vec))
            .collect(),
        decoding_results,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn decode_bit_pattern_f32<'py>(
    py: Python<'py>,
    encoding: PyRef<PyBitPatternEncoding>,
    bit_pattern_bits: HashSet<usize>,
    bit_pattern_length: usize,
    bits_per_chunk: usize,
    decoded_array_element_counts: Vec<usize>,
) -> PyResult<(Vec<OutputArr<'py, f32>>, Vec<bool>)> {
    let output_buffer = NonUniformSequence(
        decoded_array_element_counts
            .iter()
            .map(|&numel| vec![0f32; numel])
            .collect::<Vec<_>>(),
    );

    decode_bit_pattern_generic(
        py,
        encoding,
        bit_pattern_bits,
        bit_pattern_length,
        bits_per_chunk,
        output_buffer,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn decode_bit_pattern_u16<'py>(
    py: Python<'py>,
    encoding: PyRef<PyBitPatternEncoding>,
    bit_pattern_bits: HashSet<usize>,
    bit_pattern_length: usize,
    bits_per_chunk: usize,
    decoded_array_element_counts: Vec<usize>,
) -> PyResult<(Vec<OutputArr<'py, u16>>, Vec<bool>)> {
    let output_buffer = NonUniformSequence(
        decoded_array_element_counts
            .iter()
            .map(|&numel| vec![0u16; numel])
            .collect::<Vec<_>>(),
    );

    decode_bit_pattern_generic(
        py,
        encoding,
        bit_pattern_bits,
        bit_pattern_length,
        bits_per_chunk,
        output_buffer,
    )
}
