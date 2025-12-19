#![no_main]

use std::collections::HashSet;

use faultforge::encoding::secded::decode_into;
use libfuzzer_sys::{arbitrary::Arbitrary, fuzz_target, Corpus};

use faultforge::prelude::*;

#[derive(Debug, Arbitrary)]
struct Input {
    source: Vec<u8>,
    faults: Vec<usize>,
}

fuzz_target!(|input: Input| -> Corpus {
    let Input { source, faults } = input;

    if source.is_empty() {
        return Corpus::Reject;
    }

    if faults.len() > source.len() {
        return Corpus::Reject;
    }

    let mut decoded = vec![0u8; source.len()];
    let mut encoded = source.encode().unwrap();

    let mut hit = HashSet::new();
    for &fault in &faults {
        if fault >= encoded.bits_count() || !hit.insert(fault) {
            return Corpus::Reject;
        }

        encoded.flip_bit(fault);
    }

    decode_into(&mut encoded, &mut decoded).unwrap();

    Corpus::Keep
});
