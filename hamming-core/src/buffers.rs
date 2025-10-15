//! Various [`crate::bit_buffer::BitBuffer`] implementations.

pub mod limited;
mod non_uniform;
mod padded;
pub mod uniform;

pub use limited::Limited;
pub use non_uniform::NonUniformSequence;
pub use padded::PaddedBuffer;
pub use uniform::UniformSequence;
