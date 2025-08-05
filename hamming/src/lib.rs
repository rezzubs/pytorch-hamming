pub mod byte_array;
mod u64;

pub use crate::u64::Hamming64;
pub use byte_array::ByteArray;

use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn hamming(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    fn encode(input: PyReadonlyArrayDyn<f32>) -> Vec<u8> {
        let vec = input.as_array().into_iter().copied().collect::<Vec<f32>>();

        dbg!(vec
            .into_iter()
            .flat_map(|item| item.to_le_bytes())
            .collect())
    }

    Ok(())
}
