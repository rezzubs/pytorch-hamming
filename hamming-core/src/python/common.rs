//! Components that are shared between different bit-widths and types.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::wrapper::NonUniformSequence;

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
