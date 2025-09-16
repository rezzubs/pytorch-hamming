mod bit_buffer;
mod encoding;
mod python;
pub mod wrapper;

pub use bit_buffer::{BitBuffer, SizedBitBuffer};
pub use encoding::{Decodable, Encodable, Init};
use pyo3::pymodule;

/// Encoding and decoding floating point arrays with hamming codes.
///
/// Currently only chunks of 8 bytes (module u64) is supported.
#[pymodule]
mod hamming_core {
    use pyo3::pymodule;

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
        #[pymodule_export]
        use crate::python::u64::array_list_fi;
        #[pymodule_export]
        use crate::python::u64::decode_f32;
        #[pymodule_export]
        use crate::python::u64::decode_u16;
        #[pymodule_export]
        use crate::python::u64::encode_f32;
        #[pymodule_export]
        use crate::python::u64::encode_u16;
    }
}
