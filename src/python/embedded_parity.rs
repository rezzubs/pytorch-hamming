use numpy::PyReadwriteArrayDyn;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

use crate::encoding::embedded_parity::{decode, encode};

// data bits on the left and parity bits on the right.
// 0bHHHG_GGFF_FEEE_DDDC_CCBB_BAAA_HGFE_DCBA
const F32_SCHEMES: [([usize; 3], usize); 8] = [
    ([31, 30, 29], 7), // H
    ([28, 27, 26], 6), // G
    ([25, 24, 23], 5), // F
    ([22, 21, 20], 4), // E
    ([19, 18, 17], 3), // D
    ([16, 15, 14], 2), // C
    ([13, 12, 11], 1), // B
    ([10, 9, 8], 0),   // A
];

fn encode_f32(item: &mut f32) {
    for (source, dest) in F32_SCHEMES {
        encode(source, dest, item).unwrap_or_else(|err| {
            panic!("Expected {source:?} -> {dest} to be a valid scheme, got error: {err}");
        })
    }
}

fn decode_f32(item: &mut f32) {
    for (source, dest) in F32_SCHEMES {
        decode(source, dest, item).unwrap_or_else(|err| {
            panic!("Expected {source:?} -> {dest} to be a valid scheme, got error: {err}");
        })
    }
}

#[pyfunction]
pub fn embedded_parity_encode_f32(mut arr: PyReadwriteArrayDyn<f32>) -> PyResult<()> {
    arr.as_slice_mut()
        .map_err(|_| PyValueError::new_err("`arr` is not contiguous."))?
        .par_iter_mut()
        .for_each(|item| encode_f32(item));

    Ok(())
}

#[pyfunction]
pub fn embedded_parity_decode_f32(mut arr: PyReadwriteArrayDyn<f32>) -> PyResult<()> {
    arr.as_slice_mut()
        .map_err(|_| PyValueError::new_err("`arr` is not contiguous."))?
        .par_iter_mut()
        .for_each(|item| decode_f32(item));

    Ok(())
}

// data bits on the left and parity bits on the right.
// 0bDDDC_CCBB_BAAA_DCBA
const F16_SCHEMES: [([usize; 3], usize); 4] = [
    ([15, 14, 13], 3), // D
    ([12, 11, 10], 2), // C
    ([9, 8, 7], 1),    // B
    ([6, 5, 4], 0),    // A
];

fn encode_u16(item: &mut u16) {
    for (source, dest) in F16_SCHEMES {
        encode(source, dest, item).unwrap_or_else(|err| {
            panic!("Expected {source:?} -> {dest} to be a valid scheme, got error: {err}");
        })
    }
}

fn decode_u16(item: &mut u16) {
    for (source, dest) in F16_SCHEMES {
        decode(source, dest, item).unwrap_or_else(|err| {
            panic!("Expected {source:?} -> {dest} to be a valid scheme, got error: {err}");
        })
    }
}

#[pyfunction]
pub fn embedded_parity_encode_u16(mut arr: PyReadwriteArrayDyn<u16>) -> PyResult<()> {
    arr.as_slice_mut()
        .map_err(|_| PyValueError::new_err("`arr` is not contiguous."))?
        .par_iter_mut()
        .for_each(|item| encode_u16(item));

    Ok(())
}

#[pyfunction]
pub fn embedded_parity_decode_u16(mut arr: PyReadwriteArrayDyn<u16>) -> PyResult<()> {
    arr.as_slice_mut()
        .map_err(|_| PyValueError::new_err("`arr` is not contiguous."))?
        .par_iter_mut()
        .for_each(|item| decode_u16(item));

    Ok(())
}
