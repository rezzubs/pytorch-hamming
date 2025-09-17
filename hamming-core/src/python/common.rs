use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::collections::HashMap;

use crate::{wrapper::NonUniformSequence, BitBuffer};

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
    let required_padding = buffer.len() % chunk_size;
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
            "Invalid number of bits{}, expected a multiple of {}, got {}",
            index.map(|i| format!(" in array {i}")).unwrap_or("".into()),
            num_encoded_bytes,
            buffer.num_bits()
        )));
    }

    Ok(())
}
