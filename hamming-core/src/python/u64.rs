//! Functions for 64 bit data.

use std::collections::HashMap;

use crate::{wrapper::NonUniformSequence, BitBuffer, Decodable, Encodable, SizedBitBuffer};

use itertools::Itertools;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*, pyfunction, Python};

type Encoded = [u8; 9];
type EncodedOutput<'py> = Bound<'py, PyArray1<u8>>;

/// Encode an array of float32 values as an array of uint8 values.
///
/// See module docs for details.
#[pyfunction]
pub fn encode_f32<'py>(py: Python<'py>, input: PyReadonlyArray1<'py, f32>) -> EncodedOutput<'py> {
    let mut input = input.as_array().into_iter().copied().collect::<Vec<f32>>();

    // Add padding for odd length arrays
    if input.len() % 2 != 0 {
        input.push(0.);
    }

    PyArray1::from_iter(
        py,
        input
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
    input: PyReadonlyArray1<'py, u8>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, u64)> {
    let input = input.as_array();

    const NUM_ENCODED_BYTES: usize = 9;

    if input.len() % NUM_ENCODED_BYTES != 0 {
        return Err(PyValueError::new_err(format!(
            "Expected a number of bytes divisible by {NUM_ENCODED_BYTES}"
        )));
    }

    let mut iter = input.iter().copied();
    let num_encoded_buffers = input.len() / NUM_ENCODED_BYTES;
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
pub fn encode_u16<'py>(py: Python<'py>, input: PyReadonlyArray1<'py, u16>) -> EncodedOutput<'py> {
    let mut input = input.as_array().into_iter().copied().collect::<Vec<u16>>();

    const ITEMS_PER_CONTAINER: usize = 4;
    let required_padding = input.len() % ITEMS_PER_CONTAINER;
    input.extend(std::iter::repeat_n(0, required_padding));

    PyArray1::from_iter(
        py,
        input
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
    input: PyReadonlyArray1<'py, u8>,
) -> PyResult<(Bound<'py, PyArray1<u16>>, u64)> {
    let input = input.as_array();

    const NUM_ENCODED_BYTES: usize = 9;

    if input.len() % NUM_ENCODED_BYTES != 0 {
        return Err(PyValueError::new_err(format!(
            "Expected a number of bytes divisible by {NUM_ENCODED_BYTES}"
        )));
    }

    let mut iter = input.iter().copied();
    let num_encoded_buffers = input.len() / NUM_ENCODED_BYTES;
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

type FiContext = HashMap<&'static str, usize>;

#[pyfunction]
pub fn array_list_fi<'py>(
    py: Python<'py>,
    input: Vec<PyReadonlyArray1<'py, u8>>,
    ber: f64,
) -> PyResult<(Vec<EncodedOutput<'py>>, FiContext)> {
    let mut buffer = NonUniformSequence(
        input
            .into_iter()
            .map(|arr| arr.as_array().into_iter().copied().collect::<Vec<u8>>())
            .collect::<Vec<Vec<u8>>>(),
    );

    for (i, arr) in buffer.0.iter().enumerate() {
        if arr.num_bits() % Encoded::NUM_BITS != 0 {
            return Err(PyValueError::new_err(format!(
                "Invalid number of bits in array {}, expected a multiple of {}, got {}",
                i,
                Encoded::NUM_BITS,
                buffer.num_bits()
            )));
        }
    }

    let num_faults = buffer.flip_by_ber(ber);

    let context = HashMap::from([
        ("num_faults", num_faults),
        ("total_bits", buffer.num_bits()),
    ]);

    Ok((
        buffer
            .0
            .into_iter()
            .map(|arr| PyArray1::from_slice(py, &arr))
            .collect::<Vec<_>>(),
        context,
    ))
}
