use super::*;

#[test]
fn no_faults() {
    type T = [u8; 8];
    let initial: T = Init::init();
    let mut encoded = initial.encode();
    let (decoded, success): (T, bool) = encoded.decode();
    assert!(success);
    assert_eq!(initial, decoded);

    let initial: [u8; 8] = [u8::MAX; 8];
    let mut encoded = initial.encode();
    let (decoded, success): (T, bool) = encoded.decode();
    assert!(success);
    assert_eq!(initial, decoded);
}

#[test]
fn exactly_1_fault() {
    type T = [u8; 8];
    let original: T = Init::init();
    let encoded = original.encode();

    for i in 1..encoded.num_bits() {
        let mut encoded = encoded;
        encoded.flip_bit(i);
        let (decoded, success): (T, bool) = encoded.decode();
        assert!(success);
        assert_eq!(decoded, original);
    }

    // If bit 0 flips then we cannot verify that the data is correct.
    let mut encoded = encoded;
    encoded.flip_bit(0);
    let (decoded, success): (T, bool) = encoded.decode();

    assert!(!success);
    assert_eq!(decoded, original);
}

#[test]
fn exactly_2_faults() {
    type T = [u8; 8];
    let original: T = Init::init();
    let encoded = original.encode();

    for i in 1..encoded.num_bits() {
        let mut encoded = encoded;
        encoded.flip_bit(i);
        encoded.flip_bit(i - 1);
        let (_, success): (T, bool) = encoded.decode();
        assert!(!success);
    }
}

#[test]
fn auto_constants() {
    type EncodingFor8Byte = [u8; 9];
    assert_eq!(
        <EncodingFor8Byte as Decodable<[u8; 8]>>::NUM_ENCODED_BITS,
        72
    );
    assert_eq!(
        <EncodingFor8Byte as Decodable<[u8; 8]>>::NUM_ERROR_CORRECTION_BITS,
        7
    );

    type EncodingFor16Byte = [u8; 18];
    assert_eq!(
        <EncodingFor16Byte as Decodable<[u8; 16]>>::NUM_ENCODED_BITS,
        137
    );
    assert_eq!(
        <EncodingFor16Byte as Decodable<[u8; 16]>>::NUM_ERROR_CORRECTION_BITS,
        8
    );
}
