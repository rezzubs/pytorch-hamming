//! Various [`crate::bit_buffer::BitBuffer`] implementations.

mod limited;
mod non_uniform;
mod padded;

pub use limited::Limited;
pub use non_uniform::NonUniformSequence;
pub use padded::PaddedBuffer;
