pub mod byte_array;
mod u64;

pub use crate::u64::Hamming64;
pub use byte_array::ByteArray;

use itertools::Itertools;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn hamming(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    fn encode64<'py>(
        py: Python<'py>,
        input: PyReadonlyArray1<'py, f32>,
    ) -> Bound<'py, PyArray1<u8>> {
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
                .flat_map(|(a, b)| Hamming64::encode([a, b]).0.into_bytes()),
        )
    }

    // fn decode64(input: PyReadonlyArrayDyn<u8>) -> PyArray1<f32> {
    //     input.into_iter()
    // }

    Ok(())
}
