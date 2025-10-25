from __future__ import annotations

import abc
import copy
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel
from torch import nn
from typing_extensions import override

from .tensor_ops import (
    tensor_list_compare_bitwise,
    tensor_list_fault_injection,
    tensor_list_num_bits,
)

logger = logging.getLogger(__name__)


class MetaDataError(Exception):
    """The metadata didn't match"""


def count_ones(number: int) -> int:
    """Count the number of bits set to one for an integer"""
    # NOTE: `bit_count` counts ones for the absolute value. The values for this
    # are returned from rust as usize which are not expected to be negative but
    # it's best to make sure.
    assert number >= 0
    return number.bit_count()


MetaData = dict[str, str]


class BaseSystem(abc.ABC):
    @abc.abstractmethod
    def system_root_module(self) -> nn.Module:
        """Get the root module of the model.

        Other functions that operate on a `root_module` expect clones of this value.
        """

    @abc.abstractmethod
    def system_accuracy(self, root_module: nn.Module) -> float:
        """Get the accuracy of the given root_module."""

    @abc.abstractmethod
    def system_data_tensors(self, root_module: nn.Module) -> list[torch.Tensor]:
        """Return a list of references to data parameters of the root module."""

    def system_metadata(self) -> MetaData:
        """Return metadata about the system.

        This will be used to uniquely identify the system.
        """
        return MetaData()

    def system_total_num_bits(self) -> int:
        return tensor_list_num_bits(self.system_data_tensors(self.system_root_module()))


class Data(BaseModel):
    """Fault injection data for a system."""

    faults_count: int
    bits_count: int
    metadata: MetaData
    entries: list[Data.Entry]

    class Entry(BaseModel):
        """An entry corresponding a single run of fault injection."""

        accuracy: float
        faulty_parameters: list[int]

        @dataclass
        class Summary:
            accuracy: float
            faults_count: int
            bits_count: int
            # number of faulty bits per parameter: number of occurences
            n_bit_error_counts: dict[int, int]

            @override
            def __str__(self) -> str:
                error_counts_str = "\n".join(
                    f"{num_params} parameters had {num_faults} faulty bits"
                    for num_faults, num_params in self.error_counts_sorted()
                )
                return f"""
                Flipped {self.faults_count}/{self.bits_count} bits - BER: ~{self.bit_error_rate():.2e}
                Accuracy: ~{self.accuracy:.2f}%
                {self.output_faulty_parameters_count()} parameters were affected
                {self.output_faulty_bits_count()} bits were measured faulty (~{self.masked_percentage():.1f} masked)
                {error_counts_str}
                """

            def error_counts_sorted(self) -> list[tuple[int, int]]:
                """Return the `n_bit_error_counts` sorted by the number of bits"""
                counts = list(self.n_bit_error_counts.items())
                counts.sort(key=lambda x: x[0])
                return counts

            def bit_error_rate(self) -> float:
                """Return the input bit error rate for fault injection"""
                return self.faults_count / self.bits_count

            def output_faulty_parameters_count(self) -> int:
                """Count the number of parameters hit by fault injection"""
                return sum(self.n_bit_error_counts.values())

            def output_faulty_bits_count(self) -> int:
                """Count the number of bits that were actually affected by fault injection.

                This is different from the number of bits injected because encodign might mask a number of faults.
                """
                count = 0
                for num_faults, num_parameters in self.n_bit_error_counts.items():
                    count += num_faults * num_parameters
                return count

            def masked_percentage(self) -> float:
                """How many bits were masked by encoding."""
                return 1 - ((self.output_faulty_bits_count() / self.faults_count) * 100)

        def summary(self, parent: Data) -> Data.Entry.Summary:
            out = Data.Entry.Summary(
                accuracy=self.accuracy,
                bits_count=parent.bits_count,
                faults_count=parent.faults_count,
                n_bit_error_counts=dict(),
            )

            for faulty in self.faulty_parameters:
                num_invalid_bits = count_ones(faulty)
                try:
                    out.n_bit_error_counts[num_invalid_bits] += 1
                except KeyError:
                    out.n_bit_error_counts[num_invalid_bits] = 1

            return out

    def record_entry(self, system: BaseSystem, summary: bool = False) -> Data.Entry:
        """Record a new data entry for the given `system`"""

        logger.debug("Recording new data entry")

        if system.system_metadata() != self.metadata:
            raise MetaDataError(
                f"""Data has different metadata than the given system:
                data: {self.metadata}
                system: {system.system_metadata()}
                """
            )

        root = copy.deepcopy(system.system_root_module())

        data_tensors = system.system_data_tensors(root)
        original_tensors = copy.deepcopy(data_tensors)

        if self.faults_count > 0:
            logger.debug("Running fault injection")
            tensor_list_fault_injection(data_tensors, self.faults_count)
        else:
            logger.debug("Skipping fault injection")

        logger.debug("Recording accuracy")
        accuracy = system.system_accuracy(root)

        logger.debug("Comparing outputs")
        faulty_parameters = tensor_list_compare_bitwise(original_tensors, data_tensors)

        entry = Data.Entry(
            accuracy=accuracy,
            faulty_parameters=faulty_parameters,
        )

        if summary:
            print(entry.summary(self))

        self.entries.append(entry)

        return entry

    def save(self, data_path: str) -> None:
        """Save the data to `data_path`."""

        path = Path(data_path).expanduser()

        if path.exists():
            logger.info(f'Saving data to "{path}"')
        else:
            logger.info(f'Saving data to a new file at "{path}"')

        with open(path, "w") as f:
            _ = f.write(self.model_dump_json())

    @classmethod
    def load_or_create(
        cls, data_path: str, *, faults_count: int, bits_count: int, metadata: MetaData
    ) -> Data:
        """Load existing data from disk or create a new instance if it doesn't exist.

        Note: This doesn't actually create the file. For that use `save`.
        """

        path = Path(data_path).expanduser()

        if not path.exists():
            logger.warning(
                f'Didn\'t find existing data at "{path}", creating a new instance'
            )
            return cls(
                faults_count=faults_count,
                bits_count=bits_count,
                metadata=metadata,
                entries=[],
            )

        if path.is_dir():
            logger.warning(f'The path "{path}" is not a file, saving to it will fail"')

        logger.info('Loading existing data from "{path}"')

        with open(path, "r") as f:
            content = f.read()
            return Data.model_validate_json(content)
