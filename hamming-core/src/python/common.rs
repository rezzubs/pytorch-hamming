//! Components that are shared between different bit-widths and types.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::collections::HashMap;

use crate::prelude::*;
use crate::{encoding::ZeroableArray, wrapper::NonUniformSequence};

pub type OutputArr<'py, T> = Bound<'py, PyArray1<T>>;
pub type InputArr<'py, T> = PyReadonlyArray1<'py, T>;

pub type FiContext = HashMap<&'static str, usize>;
pub fn fi_context_create(num_faults: usize, num_bits: usize) -> FiContext {
    HashMap::from([("num_faults", num_faults), ("total_bits", num_bits)])
}

/// Helper for transforming numpy arrays received from python.
pub fn prep_input_array<'py, T>(input: InputArr<'py, T>) -> Vec<T>
where
    T: numpy::Element + Copy,
{
    input.as_array().into_iter().copied().collect::<Vec<T>>()
}

/// Helper for transforming lists of numpy arrays received from python.
pub fn prep_input_array_list<'py, T>(
    input: Vec<InputArr<'py, T>>,
) -> NonUniformSequence<Vec<Vec<T>>>
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

/// Helper for encoding functions that adds padding to make the `buffer` a multiple of `chunk_size`.
pub fn add_padding<T: Default + Clone>(buffer: &mut Vec<T>, chunk_size: usize) {
    let length = buffer.len();
    let required_padding = if length < chunk_size {
        chunk_size - length
    } else {
        chunk_size - buffer.len() % chunk_size
    };

    buffer.extend(std::iter::repeat_n(Default::default(), required_padding));
}

/// Helper for checking that the encoded buffers are of the correct size.
pub fn validate_encoded_array(
    buffer: &[u8],
    num_encoded_bytes: usize,
    index: Option<usize>,
) -> PyResult<()> {
    if buffer.len() % num_encoded_bytes != 0 {
        return Err(PyValueError::new_err(format!(
            "Invalid number of bytes{}, expected a multiple of {}, got {}",
            index.map(|i| format!(" in array {i}")).unwrap_or("".into()),
            num_encoded_bytes,
            buffer.len()
        )));
    }

    Ok(())
}

/// Helper for decoding into arrays of various types.
pub fn decode<'py, const NI: usize, const NO: usize, O>(
    py: Python<'py>,
    input: InputArr<'py, u8>,
) -> PyResult<(OutputArr<'py, O>, u64)>
where
    ZeroableArray<u8, NI>: Decodable<[O; NO]>,
    [O; NO]: Default,
    O: SizedBitBuffer + numpy::Element,
{
    let buffer = prep_input_array(input);

    validate_encoded_array(&buffer, NI, None)?;

    let num_encoded_buffers = buffer.len() / NI;
    let mut output: Vec<O> = Vec::with_capacity(num_encoded_buffers * NO);

    let mut iter = buffer.into_iter();
    let mut failed_decodings: u64 = 0;
    for _ in 0..num_encoded_buffers {
        let mut encoded = ZeroableArray([0u8; NI]);

        for item in encoded.0.iter_mut() {
            *item = iter.next().expect("Length is known to be a multiple of N")
        }

        let (decoded, success): ([O; NO], bool) = encoded.decode();

        if !success {
            failed_decodings = failed_decodings
                .checked_add(1)
                .expect("Unexpectedly large number of unmasked faults");
        }

        output.extend(decoded);
    }

    Ok((PyArray1::from_vec(py, output), failed_decodings))
}

/// Helper for encoding arrays of various types.
pub fn encode<'py, const NI: usize, const NO: usize, I>(
    py: Python<'py>,
    input: InputArr<'py, I>,
) -> OutputArr<'py, u8>
where
    I: numpy::Element + Copy + Default + SizedBitBuffer,
    [I; NI]: Encodable<ZeroableArray<u8, NO>, [I; NI]> + Default,
    ZeroableArray<u8, NO>: Decodable<[I; NI]> + Default,
{
    let mut buffer = prep_input_array(input);

    add_padding(&mut buffer, NI);

    let num_items = buffer.len() / NI;
    let mut output: Vec<u8> = Vec::with_capacity(num_items * NO);

    let mut iter = buffer.into_iter();
    for _ in 0..num_items {
        let mut unprotected: [I; NI] = [Default::default(); NI];

        for item in unprotected.iter_mut() {
            *item = iter
                .next()
                .expect("The length is known to be num_items * NI")
        }

        let encoded: ZeroableArray<u8, NO> = unprotected.encode();
        output.extend_from_slice(&encoded.0);
    }

    PyArray1::from_vec(py, output)
}
