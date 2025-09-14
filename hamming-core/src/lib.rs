mod bit_buffer;
pub mod conversions;
mod encoding;
mod padded_buffer;

pub use bit_buffer::{BitBuffer, SizedBitBuffer};
pub use encoding::{Decodable, Encodable, Init};
pub use padded_buffer::PaddedBuffer;

use itertools::Itertools;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};

/// Encoding and decoding floating point arrays with hamming codes.
///
/// Currently only chunks of 8 bytes (module u64) is supported.
#[pymodule]
mod hamming_core {
    use super::*;

    /// Data will be encoded in groups of 8 bytes and the corresponding encoded version will equal 9
    /// bytes. Any input array which isn't evenly divisible into 8 byte chunks will padded with extra
    /// zeros that need to be removed manually after decoding.
    ///
    /// All encoding functions take floating point arrays as inputs and return uint8 arrays. The
    /// supported input datatypes are float32 and float16.
    ///
    /// The decoding functions additionally return the number of detected faults.
    #[pymodule]
    mod u64 {
        use crate::conversions::{
            f32x2_to_le_bytes, le_bytes_to_f32x2, le_bytes_to_u16x4, u16x4_to_le_bytes,
        };

        use super::*;

        /// Encode an array of float32 values as an array of uint8 values.
        ///
        /// See module docs for details.
        #[pyfunction]
        fn encode_f32<'py>(
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
                    .flat_map(|(a, b)| f32x2_to_le_bytes([a, b]).encode()),
            )
        }

        /// Decode an array of uint8 values into an array of float32 values.
        ///
        /// Returns: (decoded_array, num_unmasked_faults)
        #[pyfunction]
        fn decode_f32<'py>(
            py: Python<'py>,
            input: PyReadonlyArray1<'py, u8>,
        ) -> PyResult<(Bound<'py, PyArray1<f32>>, u64)> {
            let input = input.as_array();

            const NUM_ENCODED_BYTES: usize = 9;

            if input.len() % NUM_ENCODED_BYTES != 0 {
                return Err(PyValueError::new_err(format!(
                    "Expected a number of bytes divisible by {NUM_ENCODED_BYTES}"
                )));
            }

            let mut iter = input.iter().copied();
            let num_encoded_buffers = input.len() / NUM_ENCODED_BYTES;
            let mut output = Vec::with_capacity(num_encoded_buffers * 2);

            let mut failed_decodings: u64 = 0;

            for _ in 0..num_encoded_buffers {
                let mut encoded: [u8; NUM_ENCODED_BYTES] =
                    iter.next_array().expect("Within bounds");

                let (decoded, success) = encoded.decode();

                if !success {
                    failed_decodings = failed_decodings
                        .checked_add(1)
                        .expect("Unexpectedly large number of unmasked faults");
                }

                let [a, b] = le_bytes_to_f32x2(decoded);
                output.push(a);
                output.push(b);
            }

            Ok((PyArray1::from_slice(py, &output), failed_decodings))
        }

        /// Encode an array of uint16 values as an array of uint8 values.
        ///
        /// This is used as a placeholder for f16 encoding because f16 is unstable in rust.
        ///
        /// See module docs for details.
        #[pyfunction]
        fn encode_u16<'py>(
            py: Python<'py>,
            input: PyReadonlyArray1<'py, u16>,
        ) -> Bound<'py, PyArray1<u8>> {
            let mut input = input.as_array().into_iter().copied().collect::<Vec<u16>>();

            const ITEMS_PER_CONTAINER: usize = 4;
            let required_padding = input.len() % ITEMS_PER_CONTAINER;
            input.extend(std::iter::repeat_n(0, required_padding));

            PyArray1::from_iter(
                py,
                input
                    .into_iter()
                    .tuples()
                    .flat_map(|(a, b, c, d)| u16x4_to_le_bytes([a, b, c, d]).encode()),
            )
        }

        /// Decode an array of uint8 values into an array of float32 values.
        ///
        /// Returns: (decoded_array, num_unmasked_faults)
        #[pyfunction]
        fn decode_u16<'py>(
            py: Python<'py>,
            input: PyReadonlyArray1<'py, u8>,
        ) -> PyResult<(Bound<'py, PyArray1<u16>>, u64)> {
            let input = input.as_array();

            const NUM_ENCODED_BYTES: usize = 9;

            if input.len() % NUM_ENCODED_BYTES != 0 {
                return Err(PyValueError::new_err(format!(
                    "Expected a number of bytes divisible by {NUM_ENCODED_BYTES}"
                )));
            }

            let mut iter = input.iter().copied();
            let num_encoded_buffers = input.len() / NUM_ENCODED_BYTES;
            let mut output = Vec::with_capacity(num_encoded_buffers * 2);

            let mut failed_decodings: u64 = 0;

            for _ in 0..num_encoded_buffers {
                let mut encoded: [u8; NUM_ENCODED_BYTES] =
                    iter.next_array().expect("Within bounds");

                let (decoded, success) = encoded.decode();

                if !success {
                    failed_decodings = failed_decodings
                        .checked_add(1)
                        .expect("Unexpectedly large number of unmasked faults");
                }

                output.extend(le_bytes_to_u16x4(decoded));
            }

            Ok((PyArray1::from_slice(py, &output), failed_decodings))
        }
    }
}
