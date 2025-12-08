#![no_main]

use std::collections::HashSet;

use hamming_core::encoding::secded::{decode_into, is_par_i};
use libfuzzer_sys::{arbitrary::Arbitrary, fuzz_target, Corpus};

use hamming_core::prelude::*;

#[derive(Debug, Arbitrary)]
struct Input {
    source: Vec<u8>,
    faults: Vec<usize>,
}

fuzz_target!(|input: Input| -> Corpus {
    let Input { source, faults } = input;

    if source.is_empty() || faults.len() > 2 {
        return Corpus::Reject;
    }

    let mut decoded = vec![0u8; source.len()];
    let mut encoded = source.encode().unwrap();

    let mut hit = HashSet::new();
    for &fault in &faults {
        if fault >= encoded.num_bits() || !hit.insert(fault) {
            return Corpus::Reject;
        }

        encoded.flip_bit(fault);
    }

    let success = decode_into(&mut encoded, &mut decoded).unwrap();

    if faults.len() <= 1 {
        if faults.iter().all(|fault| *fault != 0) {
            assert!(success);
        } else {
            assert!(!success);
        }
        assert_eq!(source, decoded);
    } else {
        assert!(!success);

        if faults.iter().all(|fault| is_par_i(*fault)) {
            assert_eq!(source, decoded);
        } else {
            assert_ne!(source, decoded);
        }
    }

    Corpus::Keep
});
