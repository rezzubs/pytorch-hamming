#![warn(clippy::must_use_candidate)]
pub mod bit_buffer;
pub mod encoding;
pub mod prelude;
mod python;
pub mod wrapper;

use pyo3::pymodule;

#[pymodule]
mod hamming_core {
    use pyo3::pymodule;

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
