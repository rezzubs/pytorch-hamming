use hamming::ByteArray;
use hamming::Hamming64;

pub fn main() {
    let num_iterations = 1_000_000;
    for _ in 0..num_iterations {
        let input_value: u64 = rand::random();
        let original = ByteArray::from(input_value);

        let encoded = Hamming64::encode(original.clone());

        let error_idx = rand::random_range::<u16, _>(0..72) as usize;
        let mut tampered = encoded.clone();
        tampered.flip_bit(error_idx);
        assert_ne!(encoded, tampered);

        let predicted_error = tampered.error_idx();
        assert_eq!(error_idx, predicted_error);

        let decoded = tampered.decode();

        assert_eq!(decoded, original);
    }
}
