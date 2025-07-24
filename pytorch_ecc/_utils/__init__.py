"""Utility functions."""

import torch


def set_bit_high(t: torch.Tensor, bit_pos: int) -> torch.Tensor:
    """Sets a specific bit to 1 in each element of an integer tensor."""
    return t | (1 << bit_pos)


def set_bit_low(t: torch.Tensor, bit_pos: int) -> torch.Tensor:
    """Sets a specific bit to 0 in each element of an integer tensor."""
    return t & (~(1 << bit_pos))


def toggle_bit(t: torch.Tensor, bit_pos: int) -> torch.Tensor:
    """Toggles a specific bit in each element of an integer tensor."""
    return t ^ (1 << bit_pos)


def get_bit(t: torch.Tensor, bit_pos: int) -> torch.Tensor:
    """Gets the value of a specific bit in each element of an integer tensor.

    Returns: 1 for high, 0 for low.
    """
    return (t >> bit_pos) & 1
