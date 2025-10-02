//! Functions for generic arrays & lists of arrays.
use crate::prelude::*;
use crate::python::common::{fi_context_create, prep_input_array, prep_input_array_list};
use numpy::PyArray1;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

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
