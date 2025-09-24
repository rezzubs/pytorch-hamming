use super::*;

mod u32 {
    use super::*;

    type EncodedU32 = ZeroableArray<u8, 5>;

    impl Encodable<EncodedU32, [u8; 4]> for [u8; 4] {}
    impl Decodable<[u8; 4]> for EncodedU32 {}

    impl Encodable<EncodedU32, [f32; 1]> for [f32; 1] {}
    impl Decodable<[f32; 1]> for EncodedU32 {}

    impl Encodable<EncodedU32, [u16; 2]> for [u16; 2] {}
    impl Decodable<[u16; 2]> for EncodedU32 {}
}

/// 8 byte data.
mod u64 {
    use super::*;

    type EncodedU64 = ZeroableArray<u8, 9>;

    impl Encodable<EncodedU64, [u8; 8]> for [u8; 8] {}
    impl Decodable<[u8; 8]> for EncodedU64 {}

    impl Encodable<EncodedU64, [f32; 2]> for [f32; 2] {}
    impl Decodable<[f32; 2]> for EncodedU64 {}

    impl Encodable<EncodedU64, [u16; 4]> for [u16; 4] {}
    impl Decodable<[u16; 4]> for EncodedU64 {}
}

/// 16 byte data
mod u128 {
    use super::*;

    type EncodedU128 = ZeroableArray<u8, 18>;

    impl Encodable<EncodedU128, [u8; 16]> for [u8; 16] {}
    impl Decodable<[u8; 16]> for EncodedU128 {}

    impl Encodable<EncodedU128, [f32; 4]> for [f32; 4] {}
    impl Decodable<[f32; 4]> for EncodedU128 {}

    impl Encodable<EncodedU128, [u16; 8]> for [u16; 8] {}
    impl Decodable<[u16; 8]> for EncodedU128 {}
}

mod u256 {
    use super::*;

    type EncodedU256 = ZeroableArray<u8, 34>;

    impl Encodable<EncodedU256, [u8; 32]> for [u8; 32] {}
    impl Decodable<[u8; 32]> for EncodedU256 {}

    impl Encodable<EncodedU256, [f32; 8]> for [f32; 8] {}
    impl Decodable<[f32; 8]> for EncodedU256 {}

    impl Encodable<EncodedU256, [u16; 16]> for [u16; 16] {}
    impl Decodable<[u16; 16]> for EncodedU256 {}
}
