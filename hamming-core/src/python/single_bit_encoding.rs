use std::cell::LazyCell;

use crate::msb_encoding::{msb_decode, msb_encode, Scheme};
use numpy::PyReadwriteArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rayon::prelude::*;

const F32_SCHEME: LazyCell<Scheme> =
    LazyCell::new(|| Scheme::for_buffer(&0f32, 30, &[0, 1]).expect("known to be correct for f32"));

#[pyfunction]
pub fn bit30_encode_f32(mut arr: PyReadwriteArrayDyn<f32>) -> PyResult<()> {
    arr.as_slice_mut()
        .map_err(|_| PyValueError::new_err("`arr` is not contiguous."))?
        .par_iter_mut()
        .for_each(|item| msb_encode(item, &F32_SCHEME).expect("The scheme is known to be correct"));

    Ok(())
}

#[pyfunction]
pub fn bit30_decode_f32(mut arr: PyReadwriteArrayDyn<f32>) -> PyResult<()> {
    arr.as_slice_mut()
        .map_err(|_| PyValueError::new_err("`arr` is not contiguous."))?
        .par_iter_mut()
        .for_each(|item| msb_decode(item, &F32_SCHEME).expect("The scheme is known to be correct"));

    Ok(())
}
