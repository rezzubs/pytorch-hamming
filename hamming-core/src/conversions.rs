pub fn f32x2_to_le_bytes(values: [f32; 2]) -> [u8; 8] {
    let a_bytes = values[0].to_le_bytes();
    let b_bytes = values[1].to_le_bytes();
    [
        a_bytes[0], a_bytes[1], a_bytes[2], a_bytes[3], b_bytes[0], b_bytes[1], b_bytes[2],
        b_bytes[3],
    ]
}

pub fn le_bytes_to_f32x2(bytes: [u8; 8]) -> [f32; 2] {
    let a_bytes = [bytes[0], bytes[1], bytes[2], bytes[3]];
    let b_bytes = [bytes[4], bytes[5], bytes[6], bytes[7]];

    [f32::from_le_bytes(a_bytes), f32::from_le_bytes(b_bytes)]
}

pub fn u16x4_to_le_bytes(values: [u16; 4]) -> [u8; 8] {
    let a_bytes = values[0].to_le_bytes();
    let b_bytes = values[1].to_le_bytes();
    let c_bytes = values[2].to_le_bytes();
    let d_bytes = values[3].to_le_bytes();

    [
        a_bytes[0], a_bytes[1], b_bytes[0], b_bytes[1], c_bytes[0], c_bytes[1], d_bytes[0],
        d_bytes[1],
    ]
}

pub fn le_bytes_to_u16x4(bytes: [u8; 8]) -> [u16; 4] {
    let a_bytes = [bytes[0], bytes[1]];
    let b_bytes = [bytes[2], bytes[3]];
    let c_bytes = [bytes[4], bytes[5]];
    let d_bytes = [bytes[6], bytes[7]];

    [
        u16::from_le_bytes(a_bytes),
        u16::from_le_bytes(b_bytes),
        u16::from_le_bytes(c_bytes),
        u16::from_le_bytes(d_bytes),
    ]
}

// TODO: add conversion tests
