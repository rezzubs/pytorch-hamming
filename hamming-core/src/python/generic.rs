//! Fault injection on generic arrays & lists of arrays.
use crate::{
    python::common::{fi_context_create, prep_input_array, prep_input_array_list},
    BitBuffer,
};
use numpy::PyArray1;
use pyo3::prelude::*;

use crate::python::common::{FiContext, InputArr, OutputArr};

#[pyfunction]
pub fn f32_array_fi<'py>(
    py: Python<'py>,
    input: InputArr<'py, f32>,
    ber: f64,
) -> PyResult<(OutputArr<'py, f32>, FiContext)> {
    let mut buffer = prep_input_array(input);

    let num_faults = buffer.flip_by_ber(ber);
    let num_bits = buffer.num_bits();

    Ok((
        PyArray1::from_vec(py, buffer),
        fi_context_create(num_faults, num_bits),
    ))
}

#[pyfunction]
pub fn f32_array_list_fi<'py>(
    py: Python<'py>,
    input: Vec<InputArr<'py, f32>>,
    ber: f64,
) -> PyResult<(Vec<OutputArr<'py, f32>>, FiContext)> {
    let mut buffer = prep_input_array_list(input);

    let num_faults = buffer.flip_by_ber(ber);
    let num_bits = buffer.num_bits();

    Ok((
        buffer
            .0
            .into_iter()
            .map(|arr| PyArray1::from_vec(py, arr))
            .collect::<Vec<_>>(),
        fi_context_create(num_faults, num_bits),
    ))
}

#[pyfunction]
pub fn u16_array_fi<'py>(
    py: Python<'py>,
    input: InputArr<'py, u16>,
    ber: f64,
) -> PyResult<(OutputArr<'py, u16>, FiContext)> {
    let mut buffer = prep_input_array(input);

    let num_faults = buffer.flip_by_ber(ber);
    let num_bits = buffer.num_bits();

    Ok((
        PyArray1::from_vec(py, buffer),
        fi_context_create(num_faults, num_bits),
    ))
}

#[pyfunction]
pub fn u16_array_list_fi<'py>(
    py: Python<'py>,
    input: Vec<InputArr<'py, u16>>,
    ber: f64,
) -> PyResult<(Vec<OutputArr<'py, u16>>, FiContext)> {
    let mut buffer = prep_input_array_list(input);

    let num_faults = buffer.flip_by_ber(ber);
    let num_bits = buffer.num_bits();

    Ok((
        buffer
            .0
            .into_iter()
            .map(|arr| PyArray1::from_vec(py, arr))
            .collect::<Vec<_>>(),
        fi_context_create(num_faults, num_bits),
    ))
}
