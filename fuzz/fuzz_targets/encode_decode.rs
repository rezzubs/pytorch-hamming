#![no_main]

use std::collections::HashSet;

use faultforge::encoding::secded::decode_into;
use libfuzzer_sys::{arbitrary::Arbitrary, fuzz_target, Corpus};

use faultforge::prelude::*;

#[derive(Debug, Arbitrary)]
pub struct Input {
    pub source: Vec<u8>,
    pub faults: Vec<usize>,
}

impl Input {
    fn is_valid(&self) -> bool {
        !self.source.is_empty() && self.faults.len() <= self.source.len()
    }
}

fuzz_target!(|input: Input| -> Corpus {
    if !input.is_valid() {
        return Corpus::Reject;
    }

    let Input { source, faults } = input;

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
