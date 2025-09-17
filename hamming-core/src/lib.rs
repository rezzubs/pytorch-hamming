mod bit_buffer;
mod encoding;
mod python;
pub mod wrapper;

pub use bit_buffer::{BitBuffer, SizedBitBuffer};
pub use encoding::{Decodable, Encodable, Init};
use pyo3::pymodule;

#[pymodule]
mod hamming_core {
    use pyo3::pymodule;

    /// Functions related to encoding 64 byte data
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

    /// Fault injection for generic arrays.
    #[pymodule]
    mod generic {
        #[pymodule_export]
        use crate::python::generic::f32_array_fi;
        #[pymodule_export]
        use crate::python::generic::f32_array_list_fi;
        #[pymodule_export]
        use crate::python::generic::u16_array_fi;
        #[pymodule_export]
        use crate::python::generic::u16_array_list_fi;
    }
}
