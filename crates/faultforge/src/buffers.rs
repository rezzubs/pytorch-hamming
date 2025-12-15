//! Various [`crate::bit_buffer::BitBuffer`] implementations.

pub mod limited;
mod non_uniform;
pub mod uniform;

pub use limited::Limited;
pub use non_uniform::NonUniformSequence;
pub use uniform::UniformSequence;
