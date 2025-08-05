use std::time::Duration;

use hamming::ByteArray;
use hamming::Hamming64;

pub fn main() {
    let mut total = Duration::ZERO;

    let num_iterations = 1_000_000;
    for _ in 0..num_iterations {
        let input_value: u64 = rand::random();
        let input = ByteArray::from(input_value);

        let start = std::time::Instant::now();
        let mut encoded = Hamming64::encode(input);
        total += start.elapsed();

        let error_idx = rand::random_range::<u16, _>(0..72) as usize;
        encoded.flip_bit(error_idx);

        let predicted_error = encoded.error_idx();
        assert_eq!(error_idx, predicted_error);
    }

    println!(
        "encoding took {total:?}, {:?} per iteration",
        total / num_iterations
    );
}
