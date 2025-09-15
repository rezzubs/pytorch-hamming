use super::*;

mod u8 {
    use super::*;

    #[test]
    fn set1() {
        let mut val = 0b0001u8;
        val.set_1(0);
        assert_eq!(val, 0b0001);

        let mut val = 0b0001u8;
        val.set_1(1);
        assert_eq!(val, 0b0011);

        let mut val = 0b0001u8;
        val.set_1(2);
        assert_eq!(val, 0b0101);

        let mut val = 0b0001u8;
        val.set_1(3);
        assert_eq!(val, 0b1001);

        let mut val = 0b0000_0001u8;
        val.set_1(7);
        assert_eq!(val, 0b1000_0001);
    }

    #[test]
    fn set0() {
        let mut val = 0b1110u8;
        val.set_0(0);
        assert_eq!(val, 0b1110);

        let mut val = 0b1110u8;
        val.set_0(1);
        assert_eq!(val, 0b1100);

        let mut val = 0b1110u8;
        val.set_0(2);
        assert_eq!(val, 0b1010);

        let mut val = 0b1110u8;
        val.set_0(3);
        assert_eq!(val, 0b0110);

        let mut val = 0b1000_0001u8;
        val.set_0(7);
        assert_eq!(val, 0b0000_0001);
    }

    #[test]
    fn is_1() {
        assert!(0b0001u8.is_1(0));
        assert!(!0b0001u8.is_1(2));
        assert!(!0b0001u8.is_1(4));
        assert!(!0b0001u8.is_1(4));
        assert!(!0b0010u8.is_1(0));
        assert!(0b0010u8.is_1(1));
    }

    #[test]
    fn flip_bit() {
        let mut val = 0b0000u8;
        val.flip_bit(0);
        assert_eq!(val, 0b0001);

        let mut val = 0b0000u8;
        val.flip_bit(1);
        assert_eq!(val, 0b0010);

        let mut val = 0b0000u8;
        val.flip_bit(3);
        assert_eq!(val, 0b1000);

        let mut val = 0b1111u8;
        val.flip_bit(0);
        assert_eq!(val, 0b1110);
    }

    #[test]
    fn total_even() {
        assert!(0b00000000u8.total_parity_is_even());
        assert!(!0b00000001u8.total_parity_is_even());
        assert!(0b00000011u8.total_parity_is_even());
        assert!(!0b00000111u8.total_parity_is_even());
        assert!(0b10000001u8.total_parity_is_even());
        assert!(!0b10010001u8.total_parity_is_even());
        assert!(0b11111111u8.total_parity_is_even());
    }

    #[test]
    fn num_1_bits() {
        assert_eq!(0b00101100u8.num_1_bits(), 3);
        assert_eq!(0b1000001u8.num_1_bits(), 2);
    }

    #[test]
    fn fault_injection() {
        let mut buf = 0u8;
        buf.flip_n_bits(1);
        assert_eq!(buf.num_1_bits(), 1);

        let mut buf = 0u8;
        buf.flip_n_bits(2);
        assert_eq!(buf.num_1_bits(), 2);

        let mut buf = 0u8;
        buf.flip_n_bits(3);
        assert_eq!(buf.num_1_bits(), 3);

        let mut buf = 0u8;
        buf.flip_n_bits(4);
        assert_eq!(buf.num_1_bits(), 4);

        let mut buf = 0u8;
        buf.flip_by_ber(1.);
        assert_eq!(buf.num_1_bits(), 8);
    }
}

mod i32 {
    use super::*;

    #[test]
    fn flip() {
        let mut buf = 0i32;
        buf.flip_bit(31);
        assert_eq!(buf, -2147483648i32);
    }
}

mod f32 {
    use super::*;

    #[test]
    fn flip() {
        let mut buf = 1f32;
        buf.flip_bit(31);
        assert_eq!(buf, -1f32);
    }
}

mod f64 {
    use super::*;

    #[test]
    fn flip() {
        let mut buf = 1f64;
        buf.flip_bit(63);
        assert_eq!(buf, -1f64);
    }
}

mod sequence {
    use super::*;

    #[test]
    fn is_1() {
        let a = [0u8, 0b110u8];
        for i in 0..=8 {
            assert!(a.is_0(i))
        }
        assert!(a.is_1(9));
        assert!(a.is_1(10));
        assert!(a.is_0(11));
    }
}
