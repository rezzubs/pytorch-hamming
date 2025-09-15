from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from torch import nn

from ._core import (
    decode_module,
    encode_module,
    nonprotected_fi,
    protected_fi,
    compare_module_bitwise,
)

__all__ = ["HammingStats"]


def num_set_bits32(number: int) -> int:
    """Count the high bits in an integer

    `number` is expected to be a 32 bit integer.
    """
    # Force an unsigned 32 bit representation. Otherwise -1 >> 1 will cause an
    # infinite loop.
    number = number & 0xFFFFFFFF
    count = 0
    while number != 0:
        if number & 1:
            count += 1
        number >>= 1

    return count


class HammingStats:
    """Statistics for a encode, inject, decode cycle."""

    def __init__(self, is_protected: bool) -> None:
        self.was_protected = is_protected
        self.accuracy: float = 0
        self.num_faults: int = 0
        self.total_bits: int = 0
        # The xors between all true and faulty parameters.
        self.non_matching_parameters: list[int] = []

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, HammingStats):
            raise ValueError(f"Can't compare HammingStats with {type(value)}")
        return (
            self.was_protected == value.was_protected
            and self.accuracy == value.accuracy
            and self.total_bits == value.total_bits
            and self.non_matching_parameters == value.non_matching_parameters
        )

    @classmethod
    def eval(
        cls,
        module: nn.Module,
        bit_error_rate: float,
        accuracy_fn: Callable[[nn.Module, bool], float],
        half: bool,
    ) -> HammingStats:
        stats = cls(True)
        if half:
            module = module.half()
        original = copy.deepcopy(module)

        encode_module(module)
        protected_fi(module, bit_error_rate=bit_error_rate, stats=stats)
        decode_module(module)

        stats.accuracy = accuracy_fn(module, half)
        stats.non_matching_parameters = compare_module_bitwise(module, original)

        return stats

    @classmethod
    def eval_noprotect(
        cls,
        module: nn.Module,
        bit_error_rate: float,
        accuracy_fn: Callable[[nn.Module, bool], float],
        half: bool,
    ) -> HammingStats:
        stats = cls(False)
        original = copy.deepcopy(module)

        nonprotected_fi(module, bit_error_rate=bit_error_rate, stats=stats)

        stats.accuracy = accuracy_fn(module, half)
        stats.non_matching_parameters = compare_module_bitwise(module, original)

        return stats

    def get_accuracy(self) -> float:
        assert self.accuracy is not None
        return self.accuracy

    def n_flips_per_param(self) -> dict[int, int]:
        """Return the number of parameters grouped by the number of bits flipped in each."""

        out = dict()
        for p in self.non_matching_parameters:
            group = num_set_bits32(p)
            assert group != 0
            prev = out.get(group, 0)
            out[group] = prev + 1

        return out

    def num_faults_in_result(self) -> int:
        return sum([num_faulty for num_faulty in self.n_flips_per_param().values()])

    def output_bit_error_rate(self) -> float:
        assert self.total_bits is not None
        return self.num_faults_in_result() / self.total_bits

    def protection_rate(self) -> float:
        """Return the ratio between the number of true faults vs injected faults.

        True faults are the ones which got through after error correction.
        """
        return 1 - (
            self.num_faults_in_result() / self.num_faults if self.num_faults > 0 else 0
        )

    def bit_error_rate(self) -> float:
        assert self.total_bits is not None
        return self.num_faults / self.total_bits

    def summary(self) -> None:
        print("Fault Injection Summary:")
        assert self.total_bits is not None
        print(
            f"  Injected {self.num_faults} faults across {self.total_bits} bits, BER: {self.bit_error_rate()}"
        )
        print(f"  Accuracy: {self.accuracy:.2f}%")

        print(
            f"  {len(self.non_matching_parameters)} parameters were messed up from injection"
        )
        param_fault_groups = list(self.n_flips_per_param().items())
        param_fault_groups.sort(key=lambda x: x[0])
        for num_faults, num_entries in param_fault_groups:
            print(f"  {num_entries} parameters had {num_faults} faults")

    def to_dict(self) -> dict:
        out = dict()

        if self.accuracy is None:
            raise ValueError("Incomplete stats, accuracy missing, run inference")

        out["accuracy"] = self.accuracy
        out["num_faults"] = self.num_faults

        if self.total_bits is None:
            raise ValueError(
                "Incomplete stats, injected_faults missing, run fault injection"
            )

        out["total_bits"] = self.total_bits
        out["was_protected"] = self.was_protected
        out["non_matching_parameters"] = self.non_matching_parameters

        return out

    @classmethod
    def from_dict(cls, obj: Any) -> HammingStats:
        was_protected = obj["was_protected"]
        if not isinstance(was_protected, bool):
            raise ValueError(f"Expected bool, got {type(was_protected)}")

        out = cls(was_protected)

        if not isinstance(obj, dict):
            raise ValueError(f"Expected a dictionary, got {type(obj)}")

        accuracy = obj["accuracy"]
        if not isinstance(accuracy, float):
            raise ValueError(f"Expected float, got {type(accuracy)}")
        out.accuracy = accuracy

        num_faults = obj["num_faults"]
        if not isinstance(num_faults, int):
            raise ValueError(f"Expected int, got {type(num_faults)}")
        out.num_faults = num_faults

        total_bits = obj["total_bits"]
        if not isinstance(total_bits, int):
            raise ValueError(f"Expected int, got {type(total_bits)}")
        out.total_bits = total_bits

        non_matching_parameters = obj["non_matching_parameters"]
        if not isinstance(non_matching_parameters, list):
            raise ValueError(f"Expected a list, got {type(non_matching_parameters)}")
        for x in non_matching_parameters:
            if not isinstance(x, int):
                raise ValueError(f"Expected int, got {type(x)}")
        out.non_matching_parameters = non_matching_parameters

        return out
