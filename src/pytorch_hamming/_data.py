from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel
from typing_extensions import override

from ._system import BaseSystem
from .tensor_ops import (
    tensor_list_compare_bitwise,
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


@dataclass
class Autosave:
    interval: int
    path: Path


class Data(BaseModel):
    """Fault injection data for a system."""

    faults_count: int
    bits_count: int
    metadata: dict[str, str]
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

                output_str = (
                    f"""{self.output_faulty_parameters_count()} parameters were affected
{self.output_faulty_bits_count()} bits were measured faulty (~{self.masked_percentage()}% masked)
"""
                    if self.faults_count > 0
                    else ""
                )

                return f"""Flipped {self.faults_count}/{self.bits_count} bits - BER: ~{self.bit_error_rate():.2e}
Accuracy: ~{self.accuracy:.2f}%
{output_str}{error_counts_str}"""

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

            def masked_percentage(self) -> float | None:
                """How many bits were masked by encoding.

                Returns None if there were no faults to begin with
                """
                if self.faults_count == 0:
                    return None
                return (1 - (self.output_faulty_bits_count() / self.faults_count)) * 100

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

    def record_entry[T](
        self,
        system: BaseSystem[T],
        *,
        summary: bool = False,
    ) -> Data.Entry:
        """Record a new data entry for the given `system`"""

        logger.debug("Recording new data entry")

        if system.system_metadata() != self.metadata:
            raise MetaDataError(
                f"""Data has different metadata than the given system:
                data: {self.metadata}
                system: {system.system_metadata()}
                """
            )

        data = copy.deepcopy(system.system_data())

        original_tensors = copy.deepcopy(system.system_data_tensors(data))

        if self.faults_count > 0:
            logger.debug("Running fault injection")
            system.system_inject_n_faults(data, self.faults_count)
            logger.debug("Fault injection finished")
        else:
            logger.debug("Skipping fault injection")

        logger.debug("Recording accuracy")
        accuracy = system.system_accuracy(data)
        logger.debug("Finished recording accuracy")

        logger.debug("Comparing outputs")
        faulty_parameters = tensor_list_compare_bitwise(
            original_tensors, system.system_data_tensors(data)
        )
        logger.debug("Finished comparing outputs")

        entry = Data.Entry(
            accuracy=accuracy,
            faulty_parameters=faulty_parameters,
        )

        if summary:
            print(entry.summary(self))

        self.entries.append(entry)

        return entry

    def record_entries[T](
        self,
        system: BaseSystem[T],
        n: int,
        *,
        summary: bool = False,
        autosave: Autosave | None,
    ):
        if n <= 0:
            raise ValueError("Expected `n` to be a positive nonzero integer")

        for i in range(n):
            i += 1
            logger.info(f"recording entry {i}/{n}")

            _ = self.record_entry(system, summary=summary)

            if autosave is not None and autosave.interval % i == 0:
                self.save(autosave.path)

    def save(self, path: Path) -> None:
        """Save the data to the given file path in json format.

        If path doesn't exist, it will create a new file with the given name.
        The parent is expected to exist.

        If the path is a directory then a file called `data.json` will be
        created in that directory .
        """

        if path.is_dir():
            path = path.joinpath("data.json")

        if path.exists():
            logger.info(f'Saving data to "{path}"')
        else:
            logger.info(f'Saving data to a new file at "{path}"')

        with open(path, "w") as f:
            _ = f.write(self.model_dump_json())

    @classmethod
    def load_or_create(
        cls,
        data_path: Path | None,
        *,
        faults_count: int,
        bits_count: int,
        metadata: dict[str, str],
    ) -> Data:
        """Load existing data from disk or create a new instance if it doesn't exist.

        Note: This doesn't actually create the file. For that use `save`.
        """

        def create():
            return cls(
                faults_count=faults_count,
                bits_count=bits_count,
                metadata=metadata,
                entries=[],
            )

        if data_path is None:
            logger.debug("Creating new data")
            return create()

        if data_path.is_dir():
            data_path = data_path.joinpath("data.json")

        if not data_path.exists():
            logger.warning(
                f'Didn\'t find existing data at "{data_path}", creating a new instance'
            )
            return create()

        logger.info('Loading existing data from "{path}"')

        with open(data_path, "r") as f:
            content = f.read()
            return Data.model_validate_json(content)
