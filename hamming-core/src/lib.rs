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
    use crate::python::generic_fault_injection::f32_array_list_fi;
    #[pymodule_export]
    use crate::python::generic_fault_injection::u16_array_list_fi;
    #[pymodule_export]
    use crate::python::generic_fault_injection::u8_array_list_fi;

    #[pymodule_export]
    use crate::python::comparison::compare_array_list_bitwise_f32;
    #[pymodule_export]
    use crate::python::comparison::compare_array_list_bitwise_u16;

    #[pymodule_export]
    use crate::python::encoding_full::encode_full_f32;
    #[pymodule_export]
    use crate::python::encoding_full::encode_full_u16;
    #[pymodule_export]
    use crate::python::encoding_full::PyFullEncoding;

    #[pymodule_export]
    use crate::python::encoding_bit_pattern::encode_bit_pattern_f32;
    #[pymodule_export]
    use crate::python::encoding_bit_pattern::encode_bit_pattern_u16;
    #[pymodule_export]
    use crate::python::encoding_bit_pattern::PyBitPatternEncoding;

    #[pymodule_export]
    use crate::python::single_bit_encoding::bit14_decode_u16;
    #[pymodule_export]
    use crate::python::single_bit_encoding::bit14_encode_u16;
    #[pymodule_export]
    use crate::python::single_bit_encoding::bit30_decode_f32;
    #[pymodule_export]
    use crate::python::single_bit_encoding::bit30_encode_f32;

    #[pymodule_export]
    use crate::python::embedded_parity::embedded_parity_encode_f32;
    #[pymodule_export]
    use crate::python::embedded_parity::embedded_parity_encode_u16;
}
