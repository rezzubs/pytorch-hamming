mod bit_buffer;
pub mod buffers;
pub mod encoding;
pub mod prelude;
mod python;
pub mod wrapper;

use pyo3::pymodule;

#[pymodule]
mod hamming_core {
    use pyo3::pymodule;

    /// Functions related to encoding 32 byte data
    #[pymodule]
    mod u32 {
        #[pymodule_export]
        use crate::python::u32::array_list_fi;
        #[pymodule_export]
        use crate::python::u32::decode_f32;
        #[pymodule_export]
        use crate::python::u32::decode_u16;
        #[pymodule_export]
        use crate::python::u32::encode_f32;
        #[pymodule_export]
        use crate::python::u32::encode_u16;
    }

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

    /// Functions related to encoding 128 byte data
    #[pymodule]
    mod u128 {
        #[pymodule_export]
        use crate::python::u128::array_list_fi;
        #[pymodule_export]
        use crate::python::u128::decode_f32;
        #[pymodule_export]
        use crate::python::u128::decode_u16;
        #[pymodule_export]
        use crate::python::u128::encode_f32;
        #[pymodule_export]
        use crate::python::u128::encode_u16;
    }

    /// Functions related to encoding 256 byte data
    #[pymodule]
    mod u256 {
        #[pymodule_export]
        use crate::python::u256::array_list_fi;
        #[pymodule_export]
        use crate::python::u256::decode_f32;
        #[pymodule_export]
        use crate::python::u256::decode_u16;
        #[pymodule_export]
        use crate::python::u256::encode_f32;
        #[pymodule_export]
        use crate::python::u256::encode_u16;
    }

    /// Functions for generic arrays & lists of arrays.
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

        #[pymodule_export]
        use crate::python::generic::compare_array_list_bitwise_f32;
        #[pymodule_export]
        use crate::python::generic::compare_array_list_bitwise_u16;
    }
}
