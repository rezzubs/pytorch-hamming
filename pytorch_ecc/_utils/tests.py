import unittest
import torch

from ecc._utils import get_bit, set_bit_low, set_bit_high, toggle_bit


def tensor(val: int) -> torch.Tensor:
    return torch.tensor([val], dtype=torch.int64)


class TestBitOps(unittest.TestCase):
    def test_set1(self):
        self.assertEqual(set_bit_high(tensor(0b0001), 0), tensor(0b0001))
        self.assertEqual(set_bit_high(tensor(0b0001), 1), tensor(0b0011))
        self.assertEqual(set_bit_high(tensor(0b0001), 2), tensor(0b0101))
        self.assertEqual(set_bit_high(tensor(0b0001), 3), tensor(0b1001))

    def test_set0(self):
        self.assertEqual(set_bit_low(tensor(0b1110), 0), tensor(0b1110))
        self.assertEqual(set_bit_low(tensor(0b1110), 1), tensor(0b1100))
        self.assertEqual(set_bit_low(tensor(0b1110), 2), tensor(0b1010))
        self.assertEqual(set_bit_low(tensor(0b1110), 3), tensor(0b0110))
        self.assertEqual(set_bit_low(tensor(0b1000_0001), 7), tensor(0b0000_0001))

    def test_get(self):
        self.assertEqual(get_bit(tensor(0b0001), 0), tensor(1))
        self.assertEqual(get_bit(tensor(0b0001), 2), tensor(0))
        self.assertEqual(get_bit(tensor(0b0001), 4), tensor(0))
        self.assertEqual(get_bit(tensor(0b0010), 0), tensor(0))
        self.assertEqual(get_bit(tensor(0b0010), 1), tensor(1))
        self.assertEqual(get_bit(tensor(0b0010), 2), tensor(0))

    def test_toggle(self):
        self.assertEqual(toggle_bit(tensor(0b0000), 0), tensor(0b0001))
        self.assertEqual(toggle_bit(tensor(0b0000), 1), tensor(0b0010))
        self.assertEqual(toggle_bit(tensor(0b0000), 3), tensor(0b1000))
        self.assertEqual(toggle_bit(tensor(0b1111), 0), tensor(0b1110))
        self.assertEqual(toggle_bit(tensor(0b1111), 2), tensor(0b1011))
