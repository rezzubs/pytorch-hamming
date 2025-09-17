//! Functions for 64 bit data.

use crate::{
    python::common::{
        add_padding, fi_context_create, prep_input_array, prep_input_array_list,
        validate_encoded_array, FiContext, InputArr, OutputArr,
    },
    BitBuffer, Decodable, Encodable,
};

use itertools::Itertools;
use numpy::PyArray1;
use pyo3::prelude::*;

const NUM_ENCODED_BYTES: usize = 9;

/// Encode an array of float32 values as an array of uint8 values.
///
/// See module docs for details.
#[pyfunction]
pub fn encode_f32<'py>(py: Python<'py>, input: InputArr<'py, f32>) -> OutputArr<'py, u8> {
    let mut buffer = prep_input_array(input);

    add_padding(&mut buffer, 2);

    PyArray1::from_iter(
        py,
        buffer
            .into_iter()
            .tuples()
            .flat_map(|(a, b)| [a, b].encode()),
    )
}

/// Decode an array of uint8 values into an array of float32 values.
///
/// Returns: (decoded_array, num_unmasked_faults)
#[pyfunction]
pub fn decode_f32<'py>(
    py: Python<'py>,
    input: InputArr<'py, u8>,
) -> PyResult<(OutputArr<'py, f32>, u64)> {
    let buffer = prep_input_array(input);

    validate_encoded_array(&buffer, NUM_ENCODED_BYTES, None)?;

    let mut iter = buffer.iter().copied();
    let num_encoded_buffers = buffer.len() / NUM_ENCODED_BYTES;
    let mut output = Vec::with_capacity(num_encoded_buffers * 2);

    let mut failed_decodings: u64 = 0;

    for _ in 0..num_encoded_buffers {
        let mut encoded: [u8; NUM_ENCODED_BYTES] = iter.next_array().expect("Within bounds");

        let ([a, b], success) = encoded.decode();

        if !success {
            failed_decodings = failed_decodings
                .checked_add(1)
                .expect("Unexpectedly large number of unmasked faults");
        }

        output.push(a);
        output.push(b);
    }

    Ok((PyArray1::from_slice(py, &output), failed_decodings))
}

/// Encode an array of uint16 values as an array of uint8 values.
///
/// This is used as a placeholder for f16 encoding because f16 is unstable in rust.
///
/// See module docs for details.
#[pyfunction]
pub fn encode_u16<'py>(py: Python<'py>, input: InputArr<'py, u16>) -> OutputArr<'py, u8> {
    let mut buffer = prep_input_array(input);

    add_padding(&mut buffer, 4);

    PyArray1::from_iter(
        py,
        buffer
            .into_iter()
            .tuples()
            .flat_map(|(a, b, c, d)| [a, b, c, d].encode()),
    )
}

/// Decode an array of uint8 values into an array of float32 values.
///
/// Returns: (decoded_array, num_unmasked_faults)
#[pyfunction]
pub fn decode_u16<'py>(
    py: Python<'py>,
    input: InputArr<'py, u8>,
) -> PyResult<(OutputArr<'py, u16>, u64)> {
    let buffer = prep_input_array(input);

    validate_encoded_array(&buffer, NUM_ENCODED_BYTES, None)?;

    let mut iter = buffer.iter().copied();
    let num_encoded_buffers = buffer.len() / NUM_ENCODED_BYTES;
    let mut output = Vec::with_capacity(num_encoded_buffers * 4);

    let mut failed_decodings: u64 = 0;

    for _ in 0..num_encoded_buffers {
        let mut encoded: [u8; NUM_ENCODED_BYTES] = iter.next_array().expect("Within bounds");

        let (decoded, success): ([u16; 4], bool) = encoded.decode();

        if !success {
            failed_decodings = failed_decodings
                .checked_add(1)
                .expect("Unexpectedly large number of unmasked faults");
        }

        output.extend(decoded);
    }

    Ok((PyArray1::from_slice(py, &output), failed_decodings))
}

#[pyfunction]
pub fn array_list_fi<'py>(
    py: Python<'py>,
    input: Vec<InputArr<'py, u8>>,
    ber: f64,
) -> PyResult<(Vec<OutputArr<'py, u8>>, FiContext)> {
    let mut buffer = prep_input_array_list(input);

    for (i, arr) in buffer.0.iter().enumerate() {
        validate_encoded_array(arr, NUM_ENCODED_BYTES, Some(i))?;
    }

    let num_faults = buffer.flip_by_ber(ber);
    let num_bits = buffer.num_bits();

    Ok((
        buffer
            .0
            .into_iter()
            .map(|arr| PyArray1::from_slice(py, &arr))
            .collect::<Vec<_>>(),
        fi_context_create(num_faults, num_bits),
    ))
}
