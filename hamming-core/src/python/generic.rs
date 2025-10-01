//! Functions for generic arrays & lists of arrays.
use crate::{
    python::common::{fi_context_create, prep_input_array, prep_input_array_list},
    BitBuffer,
};
use numpy::PyArray1;
use pyo3::{exceptions::PyValueError, prelude::*};

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

/// Computes bit masks for non-matching bits for all items.
#[pyfunction]
pub fn compare_array_list_bitwise_f32<'py>(
    _py: Python<'py>,
    a: Vec<InputArr<f32>>,
    b: Vec<InputArr<f32>>,
) -> PyResult<Vec<u32>> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err(
            "The lengths of `a` and `b` don't match",
        ));
    }

    let output = a
        .into_iter()
        .zip(b)
        .map(|(a_inner, b_inner)| -> PyResult<Vec<u32>> {
            let a_arr = a_inner.as_array();
            let b_arr = b_inner.as_array();

            if a_arr.len() != b_arr.len() {
                return Err(PyValueError::new_err(
                    "`a` and `b` don't contain the same number of items.",
                ));
            }

            let mut output: Vec<u32> = Vec::with_capacity(a_arr.len());
            for (a_item, b_item) in a_arr.iter().zip(b_arr.iter()) {
                output.push(a_item.to_bits() | b_item.to_bits());
            }

            Ok(output)
        })
        .collect::<PyResult<Vec<Vec<u32>>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<u32>>();

    Ok(output)
}
