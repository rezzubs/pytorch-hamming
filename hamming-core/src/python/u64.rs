//! Functions for 64 bit data.

use crate::{
    prelude::*,
    python::common::{
        decode, encode, fi_context_create, prep_input_array_list, validate_encoded_array,
        FiContext, InputArr, OutputArr,
    },
};

use numpy::PyArray1;
use pyo3::prelude::*;

/// Bytes used for the **storage** of the encoded format.
const NUM_ENCODED_BYTES: usize = 9;

const FITS_F32: usize = 2;
const FITS_U16: usize = 4;

/// Encode an array of float32 values as an array of uint8 values.
///
/// See module docs for details.
#[pyfunction]
pub fn encode_f32<'py>(py: Python<'py>, input: InputArr<'py, f32>) -> OutputArr<'py, u8> {
    encode::<FITS_F32, NUM_ENCODED_BYTES, f32>(py, input)
}

/// Decode an array of uint8 values into an array of float32 values.
///
/// Returns: (decoded_array, num_unmasked_faults)
#[pyfunction]
pub fn decode_f32<'py>(
    py: Python<'py>,
    input: InputArr<'py, u8>,
) -> PyResult<(OutputArr<'py, f32>, u64)> {
    decode::<NUM_ENCODED_BYTES, FITS_F32, f32>(py, input)
}

/// Encode an array of uint16 values as an array of uint8 values.
///
/// This is used as a placeholder for f16 encoding because f16 is unstable in rust.
///
/// See module docs for details.
#[pyfunction]
pub fn encode_u16<'py>(py: Python<'py>, input: InputArr<'py, u16>) -> OutputArr<'py, u8> {
    encode::<FITS_U16, NUM_ENCODED_BYTES, u16>(py, input)
}

/// Decode an array of uint8 values into an array of float32 values.
///
/// Returns: (decoded_array, num_unmasked_faults)
#[pyfunction]
pub fn decode_u16<'py>(
    py: Python<'py>,
    input: InputArr<'py, u8>,
) -> PyResult<(OutputArr<'py, u16>, u64)> {
    decode::<NUM_ENCODED_BYTES, FITS_U16, u16>(py, input)
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
