pub mod byte_array;
mod u64;

pub use crate::u64::Hamming64;
pub use byte_array::ByteArray;

use itertools::Itertools;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};

/// A Python module implemented in Rust.
#[pymodule]
fn hamming(m: &Bound<'_, PyModule>) -> PyResult<()> {
    /// Encode an array of float32 values as an array of uint8 values.
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

    /// Decode an array of uint8 values into an array of float32 values.
    ///
    /// Returns: (decoded_array, num_unmasked_faults)
    #[pyfn(m)]
    fn decode64<'py>(
        py: Python<'py>,
        input: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, u64)> {
        let input = input.as_array();

        if input.len() % Hamming64::NUM_BYTES != 0 {
            return Err(PyValueError::new_err(format!(
                "Expected a number of bytes divisible by {}",
                Hamming64::NUM_BYTES
            )));
        }

        let mut iter = input.iter().copied();
        let num_encoded = input.len() / Hamming64::NUM_BYTES;
        let mut output = Vec::with_capacity(num_encoded * 2);

        let mut failed_decodings: u64 = 0;

        for _ in 0..num_encoded {
            let bytes: [u8; Hamming64::NUM_BYTES] = iter.next_array().expect("Within bounds");

            let encoded = Hamming64(bytes.into());
            let (decoded, success) = encoded.decode();

            if !success {
                failed_decodings = failed_decodings
                    .checked_add(1)
                    .expect("Unexpectedly large number of unmasked faults");
            }

            let [a, b]: [f32; 2] = decoded.into();
            output.push(a);
            output.push(b);
        }

        Ok((PyArray1::from_slice(py, &output), failed_decodings))
    }

    Ok(())
}
