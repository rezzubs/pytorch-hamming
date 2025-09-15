use super::*;

/// 8 byte data.
mod u64 {
    use super::*;

    type EncodedU64 = [u8; 9];

    impl Init for EncodedU64 {
        fn init() -> Self {
            Default::default()
        }
    }

    impl Init for [u8; 8] {
        fn init() -> Self {
            Default::default()
        }
    }

    impl Encodable<EncodedU64, [u8; 8]> for [u8; 8] {}
    impl Decodable<[u8; 8]> for EncodedU64 {}

    impl Init for [f32; 2] {
        fn init() -> Self {
            Default::default()
        }
    }

    impl Encodable<EncodedU64, [f32; 2]> for [f32; 2] {}
    impl Decodable<[f32; 2]> for EncodedU64 {}

    impl Init for [u16; 4] {
        fn init() -> Self {
            Default::default()
        }
    }

    impl Encodable<EncodedU64, [u16; 4]> for [u16; 4] {}
    impl Decodable<[u16; 4]> for EncodedU64 {}
}

/// 16 byte data
mod u128 {
    use super::*;

    // TODO: PaddedBuffer usage.

    impl Init for [u8; 16] {
        fn init() -> Self {
            Default::default()
        }
    }

    impl Init for [u8; 18] {
        fn init() -> Self {
            Default::default()
        }
    }

    impl Encodable<[u8; 18], [u8; 16]> for [u8; 16] {}
    impl Decodable<[u8; 16]> for [u8; 18] {}
}

mod u256 {
    use super::*;

    impl Init for [u8; 32] {
        fn init() -> Self {
            Default::default()
        }
    }

    impl Init for [u8; 34] {
        fn init() -> Self {
            [0u8; 34]
        }
    }

    impl Encodable<[u8; 34], [u8; 32]> for [u8; 32] {}
    impl Decodable<[u8; 32]> for [u8; 34] {}
}
