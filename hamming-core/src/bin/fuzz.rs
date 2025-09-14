use hamming_core::{BitBuffer, Decodable, Encodable};
use rand::seq::IteratorRandom;

pub fn main() {
    let num_iterations = 1_000_000;
    single_bitflip(num_iterations);
    double_bitflip(num_iterations);
}

fn single_bitflip(num_iterations: usize) {
    for _ in 0..num_iterations {
        let input_value: u64 = rand::random();
        let original = input_value.to_le_bytes();

        let encoded = original.encode();

        let error_index = rand::random_range::<u16, _>(0..72) as usize;
        let mut tampered = encoded;
        tampered.flip_bit(error_index);
        assert_ne!(encoded, tampered);

        let predicted_error = tampered.error_index();
        assert_eq!(error_index, predicted_error);

        let (decoded, success) = tampered.decode();

        if error_index == 0 {
            assert!(!success);
        } else {
            assert!(success);
        }

        assert_eq!(decoded, original);
    }
}

fn double_bitflip(num_iterations: usize) {
    for _ in 0..num_iterations {
        let input_value: u64 = rand::random();
        let original = input_value.to_le_bytes();

        let encoded = original.encode();

        let errors = (0..72usize).choose_multiple(&mut rand::rng(), 2);
        assert_eq!(errors.len(), 2);
        assert_ne!(errors[0], errors[1]);

        let mut tampered = encoded;
        tampered.flip_bit(errors[0]);
        tampered.flip_bit(errors[1]);
        assert_ne!(encoded, tampered);

        let predicted_error = tampered.error_index();
        if errors.iter().all(|x| *x != 0) {
            // NOTE: if one of the flips is 0 then the prediction will be the other flip.
            assert_ne!(predicted_error, errors[0]);
            assert_ne!(predicted_error, errors[1]);
        }

        let (_, success) = tampered.decode();
        assert!(!success);
    }
}
