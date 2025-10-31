#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![warn(clippy::must_use_candidate)]
pub mod bit_buffer;
pub mod buffers;
pub mod encoding;
pub mod prelude;
mod python;

use pyo3::pymodule;

#[pymodule]
mod hamming_core {
    #[pymodule_export]
    use crate::python::f32_array_list_fi;
    #[pymodule_export]
    use crate::python::u16_array_list_fi;
    #[pymodule_export]
    use crate::python::u8_array_list_fi;

    #[pymodule_export]
    use crate::python::compare_array_list_bitwise_f32;
    #[pymodule_export]
    use crate::python::compare_array_list_bitwise_u16;

    #[pymodule_export]
    use crate::python::encode_full_f32;
    #[pymodule_export]
    use crate::python::encode_full_u16;

    #[pymodule_export]
    use crate::python::decode_full_f32;
    #[pymodule_export]
    use crate::python::decode_full_u16;

    #[pymodule_export]
    use crate::python::encode_bit_pattern_f32;
    #[pymodule_export]
    use crate::python::encode_bit_pattern_u16;
}
