//! Most significant bit protection for bit buffers.
//!
//! A choosen bit is copied into an odd number of other bits (see [`Scheme`]). When decoding, a
//! majority vote will be used for the final value of the protected bit.

use std::collections::HashSet;

use crate::prelude::*;

/// A scheme for most significant bit encoding.
///
/// See [`msb_encode`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scheme {
    /// The length of the buffer that the scheme was configured for.
    buffer_length: usize,
    /// The source bit that's going to be copied into targets.
    source: usize,
    /// The duplicates of `source`.
    targets: Box<[usize]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SchemeCreationError {
    /// The were an odd number of target bits
    #[error("The were an odd number of target bits")]
    OddTargets,
    /// The list of targets was empty
    #[error("The list of targets was empty")]
    EmptyTargets,
    /// The list of targets contained duplicate entries
    #[error("Index {0} occurs more than once.")]
    DuplicateIndex(usize),
    /// An index was out of bounds for the buffer
    #[error("The index {0} is out of bounds for the buffer")]
    IndexOutOfBounds(usize),
}

impl Scheme {
    /// Create a new scheme
    pub fn new(
        buffer_length: usize,
        from: usize,
        targets: &[usize],
    ) -> Result<Self, SchemeCreationError> {
        if targets.is_empty() {
            return Err(SchemeCreationError::EmptyTargets);
        }

        if !targets.len().is_multiple_of(2) {
            return Err(SchemeCreationError::OddTargets);
        }

        for &i in targets.iter().chain([&from]) {
            if i >= buffer_length {
                return Err(SchemeCreationError::IndexOutOfBounds(i));
            }
        }

        if let Err(non_unique) = is_unique(targets.iter().chain([&from])) {
            return Err(SchemeCreationError::DuplicateIndex(*non_unique));
        }

        Ok(Self {
            buffer_length,
            source: from,
            targets: targets.into(),
        })
    }

    pub fn for_buffer<B>(
        buffer: &B,
        from: usize,
        targets: &[usize],
    ) -> Result<Self, SchemeCreationError>
    where
        B: BitBuffer,
    {
        Self::new(buffer.num_bits(), from, targets)
    }

    pub fn is_valid_for<B>(&self, buffer: &B) -> bool
    where
        B: BitBuffer,
    {
        self.buffer_length == buffer.num_bits()
    }
}

/// The scheme is not configured for a buffer of this length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("The scheme is not configured for a buffer of this length")]
pub struct InvalidSchemeError;

/// Duplicate a bit into a number of other bits.
///
/// The buffer is not modified when the scheme is invalid.
///
/// See module docs for details.
pub fn msb_encode<B>(buffer: &mut B, scheme: &Scheme) -> Result<(), InvalidSchemeError>
where
    B: BitBuffer,
{
    if !scheme.is_valid_for(buffer) {
        return Err(InvalidSchemeError);
    }

    let source_is_1 = buffer.is_1(scheme.source);

    for &target in &scheme.targets {
        if source_is_1 {
            buffer.set_1(target);
        } else {
            buffer.set_0(target);
        }
    }

    Ok(())
}

/// Decode the buffer with this scheme.
///
/// The buffer is not modified when the scheme is invalid.
///
/// See module docs for details.
pub fn msb_decode<B>(buffer: &mut B, scheme: &Scheme) -> Result<(), InvalidSchemeError>
where
    B: BitBuffer,
{
    if !scheme.is_valid_for(buffer) {
        return Err(InvalidSchemeError);
    }

    let indices = scheme.targets.iter().copied().chain([scheme.source]);

    let mut is_1 = 0;
    let mut is_0 = 0;

    for i in indices {
        if buffer.is_1(i) {
            is_1 += 1
        } else {
            is_0 += 1
        }
    }

    // Cannot be the same for a valid scheme because the number of `indices` is always odd.
    #[cfg(debug_assertions)]
    if (is_1 != 0) && is_0 != 0 {
        debug_assert_ne!(is_1, is_0);
    }

    if is_1 > is_0 {
        buffer.set_1(scheme.source);
    } else {
        buffer.set_0(scheme.source);
    }

    Ok(())
}

fn is_unique<T>(iter: impl IntoIterator<Item = T>) -> Result<(), T>
where
    T: std::hash::Hash + std::cmp::Eq + Copy,
{
    let mut seen = HashSet::new();

    for item in iter {
        if !seen.insert(item) {
            return Err(item);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_schemes() {
        let buffer = 0u32;

        assert_eq!(
            Scheme::for_buffer(&buffer, 32, &[0, 1]),
            Err(SchemeCreationError::IndexOutOfBounds(32))
        );
        assert_eq!(
            Scheme::for_buffer(&buffer, 31, &[0, 32]),
            Err(SchemeCreationError::IndexOutOfBounds(32))
        );
        assert_eq!(
            Scheme::for_buffer(&buffer, 31, &[0, 32]),
            Err(SchemeCreationError::IndexOutOfBounds(32))
        );

        assert_eq!(
            Scheme::for_buffer(&buffer, 31, &[31, 1]),
            Err(SchemeCreationError::DuplicateIndex(31))
        );
        assert_eq!(
            Scheme::for_buffer(&buffer, 31, &[0, 0]),
            Err(SchemeCreationError::DuplicateIndex(0))
        );

        assert_eq!(
            Scheme::for_buffer(&buffer, 31, &[0, 1, 2]),
            Err(SchemeCreationError::OddTargets)
        );
        assert_eq!(
            Scheme::for_buffer(&buffer, 31, &[0]),
            Err(SchemeCreationError::OddTargets)
        );

        assert_eq!(
            Scheme::for_buffer(&buffer, 31, &[]),
            Err(SchemeCreationError::EmptyTargets)
        );

        let mut buffer = 0u32;
        assert_eq!(
            msb_encode(&mut buffer, &Scheme::new(33, 31, &[32, 0]).unwrap()),
            Err(InvalidSchemeError)
        );
    }

    #[test]
    fn encode_decode_2copies() {
        let buffers = (0..=u16::MAX).collect::<Vec<_>>();

        let source_bit = 15;
        let scheme = Scheme::for_buffer(&0u16, source_bit, &[0, 1]).unwrap();

        for original in buffers {
            let mut encoded = original;

            msb_encode(&mut encoded, &scheme).unwrap();

            let mut decoded = encoded;

            // Decode without faults
            msb_decode(&mut decoded, &scheme).unwrap();

            assert_eq!(encoded, decoded);

            for fault_index in [0, 1, source_bit] {
                let mut faulty = encoded;

                faulty.flip_bit(fault_index);
                assert_ne!(faulty, encoded);

                msb_decode(&mut faulty, &scheme).unwrap();

                assert_eq!(faulty.is_1(source_bit), encoded.is_1(source_bit));
            }
        }
    }

    #[test]
    fn encode_decode_4copies() {
        let buffers = (0..=u16::MAX).collect::<Vec<_>>();

        let source_bit = 15;
        let scheme = Scheme::for_buffer(&0u16, source_bit, &[0, 1, 2, 3]).unwrap();

        for original in buffers {
            let mut encoded = original;

            msb_encode(&mut encoded, &scheme).unwrap();

            let mut decoded = encoded;

            // Decode without faults
            msb_decode(&mut decoded, &scheme).unwrap();

            assert_eq!(encoded, decoded);

            let faults = [0, 1, 2, 3, source_bit];

            for (&fault1, fault2) in faults.iter().zip(faults) {
                if fault1 == fault2 {
                    continue;
                }
                let mut faulty = encoded;

                faulty.flip_bit(fault1);
                faulty.flip_bit(fault2);

                assert_ne!(faulty, encoded);

                msb_decode(&mut faulty, &scheme).unwrap();

                assert_eq!(faulty.is_1(source_bit), encoded.is_1(source_bit));
            }
        }
    }
}
