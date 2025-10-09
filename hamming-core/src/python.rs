//! Python bindings

mod common;
pub mod generic;

macro_rules! bufer_impl {
    ($ident:ident, $num_original:expr, $num_encoded:expr) => {
        pub mod $ident {
            use crate::{
                encoding::ZeroableArray,
                prelude::*,
                python::common::{
                    decode, encode, fi_context_create, prep_input_array_list,
                    validate_encoded_array, FiContext, InputArr, OutputArr,
                },
                wrapper::PaddedBuffer,
            };

            use numpy::PyArray1;
            use pyo3::prelude::*;

            const NUM_ORIGINAL_BYTES: usize = $num_original;
            /// Bytes used for the **storage** of the encoded format.
            const NUM_ENCODED_BYTES: usize = $num_encoded;

            const FITS_F32: usize = (NUM_ORIGINAL_BYTES * 8) / 32;
            const FITS_U16: usize = FITS_F32 * 2;

            type Original = [u8; NUM_ORIGINAL_BYTES];
            type Encoded = ZeroableArray<u8, NUM_ENCODED_BYTES>;

            const NUM_PADDING_BITS: usize = <Encoded as Decodable<Original>>::NUM_PADDING_BITS;
            const NUM_ENCODED_BITS: usize = <Encoded as Decodable<Original>>::NUM_ENCODED_BITS;

            /// Encode an array of float32 values as an array of uint8 values.
            ///
            /// See module docs for details.
            #[pyfunction]
            pub fn encode_f32<'py>(
                py: Python<'py>,
                input: InputArr<'py, f32>,
            ) -> OutputArr<'py, u8> {
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
            pub fn encode_u16<'py>(
                py: Python<'py>,
                input: InputArr<'py, u16>,
            ) -> OutputArr<'py, u8> {
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
                let buffer = prep_input_array_list(input);

                for (i, arr) in buffer.0.iter().enumerate() {
                    validate_encoded_array(arr, NUM_ENCODED_BYTES, Some(i))?;
                }

                let mut buffer: PaddedBuffer<_, NUM_ENCODED_BITS, NUM_PADDING_BITS> =
                    PaddedBuffer::new(buffer);

                let num_faults = buffer.flip_by_ber(ber);
                let num_bits = buffer.num_bits();

                Ok((
                    buffer
                        .into_inner()
                        .0
                        .into_iter()
                        .map(|arr| PyArray1::from_slice(py, &arr))
                        .collect::<Vec<_>>(),
                    fi_context_create(num_faults, num_bits),
                ))
            }
        }
    };
}

// u64 is a manual implementation because it doesn't use `PaddedBuffer`.
pub mod u64;

bufer_impl!(u32, 4, 5);
bufer_impl!(u128, 16, 18);
bufer_impl!(u256, 32, 34);
