from __future__ import annotations

import copy
import functools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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


def metadata_str(metadata: dict[str, str], bit_error_rate: float) -> str:
    parts = list(metadata.items())
    parts.sort(key=lambda x: x[0])

    parts_strs = ["-".join(p) for p in parts]
    parts_strs.append(f"ber-{bit_error_rate:.2e}")

    return "_".join(parts_strs)


def get_path(
    root: Path, bit_error_rate: float, metadata: dict[str, str], metadata_name: bool
) -> Path:
    if (not root.exists()) and metadata_name:
        logger.info(f"Creating a new output directory at {root}")
        root.mkdir()

    if root.is_dir():
        if metadata_name:
            logger.debug("generating file name from metadata")
            root = root.joinpath(metadata_str(metadata, bit_error_rate) + ".json")
        else:
            logger.debug("metadata_name not set, defaulting to data.json")
            root = root.joinpath("data.json")
    elif metadata_name:
        raise ValueError(
            "`metadata_name` can only be used together with directory paths"
        )

    return root


@dataclass
class Autosave:
    interval: int
    path: Path
    metadata_name: bool


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
{self.output_faulty_bits_count()} bits were measured faulty ({self.masked_percentage():.2f}% masked)
"""
                    if self.faults_count > 0
                    else ""
                )

                return f"""Flipped {self.faults_count}/{self.bits_count} bits - BER: {self.bit_error_rate():.2e}
Accuracy: {self.accuracy:.2f}%
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

        def faults_per_bit_index(
            self, skip_multi_bit_faults: bool = False
        ) -> dict[int, int]:
            """Get the number of faults for each bit index.

            The keys of the dictionary map to the indices and the value to the
            number of faults.
            """
            index_map: dict[int, int] = dict()

            for fault_mask in self.faulty_parameters:
                binary_str = bin(fault_mask)
                if skip_multi_bit_faults:
                    if binary_str.count("1") > 1:
                        continue

                for i, char in enumerate(reversed(binary_str)):
                    if char == "1":
                        index_map[i] = 1 + index_map.get(i, 0)

            return index_map

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

        data = system.system_clone_data(system.system_data())

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
        autosave: Autosave | None = None,
    ):
        logger.debug(f"Recording {n} runs")
        if n <= 0:
            raise ValueError("Expected `n` to be a positive nonzero integer")

        for i in range(n):
            i += 1
            logger.info(f"recording entry {i}/{n}")

            _ = self.record_entry(system, summary=summary)

            if autosave is not None and i % autosave.interval == 0:
                self.save(autosave.path, autosave.metadata_name)

    def record_until_stable[T](
        self,
        system: BaseSystem[T],
        *,
        threshold: float,
        stable_within: int,
        min_runs: int | None = None,
        summary: bool = False,
        autosave: Autosave | None = None,
    ):
        logger.debug(
            f"Recording until mean is within {threshold}% in the last {stable_within} cycles"
        )
        if min_runs is None or min_runs < stable_within:
            min_runs = stable_within

        if stable_within <= 0:
            raise ValueError("`stable_within` must be greater than 0")

        autosave_counter = 0
        while len(self.entries) < min_runs:
            autosave_counter += 1
            logger.info(f"Recording run {len(self.entries) + 1}/{min_runs}min")

            _ = self.record_entry(system, summary=summary)

            if autosave is not None:
                rem = autosave_counter % autosave.interval
                if rem == 0:
                    logger.debug("autosave triggered")
                    self.save(autosave.path, autosave.metadata_name)
                else:
                    remaining = autosave.interval - rem
                    logger.debug(f"{remaining} runs until autosave")

        logger.info(f"Passed the minimum number of runs ({min_runs})")

        while not self.is_stable(stable_within, threshold):
            autosave_counter += 1
            drift = self.mean_drift(stable_within)
            assert drift is not None, (
                "Has to be a real value after the minimum number of runs"
            )
            drift_min, drift_max = drift

            logger.info(
                f"Recording run {len(self.entries)} to achieve stability at {threshold:.3}%, currently at {drift_max - drift_min:.3}%"
            )

            _ = self.record_entry(system, summary=summary)

            if autosave is not None:
                rem = autosave_counter % autosave.interval
                if rem == 0:
                    logger.debug("autosave triggered")
                    self.save(autosave.path, autosave.metadata_name)
                else:
                    remaining = autosave.interval - rem
                    logger.debug(f"{remaining} runs until autosave")

        logger.info("Data mean is stable")

    def save(self, data_path: Path, metadata_name: bool = False) -> None:
        """Save the data to the given file path in json format.

        If path doesn't exist, it will create a new file with the given name.
        The parent is expected to exist.

        If the path is a directory then a file called `data.json` will be
        created in that directory.

        If `metadata_path` is True then the file name is set based on the
        metadata and bit error rate. `data_path` must be a directory in this
        case.
        """

        data_path = get_path(
            data_path, self.faults_count / self.bits_count, self.metadata, metadata_name
        )

        if data_path.exists():
            logger.info(f'Saving data to "{data_path}"')
        else:
            logger.info(f'Saving data to a new file at "{data_path}"')

        with open(data_path, "w") as f:
            _ = f.write(self.model_dump_json())

    @classmethod
    def load(
        cls,
        data_path: Path,
    ) -> Data:
        with open(data_path, "r") as f:
            content = f.read()
            return Data.model_validate_json(content)

    @classmethod
    def load_or_create(
        cls,
        data_path: Path | None,
        *,
        faults_count: int,
        bits_count: int,
        metadata: dict[str, str],
        metadata_name: bool = False,
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

        data_path = get_path(
            data_path, faults_count / bits_count, metadata, metadata_name
        )

        if not data_path.exists():
            logger.warning(
                f'Didn\'t find existing data at "{data_path}", creating a new instance'
            )
            return create()

        logger.info('Loading existing data from "{path}"')

        return cls.load(
            data_path,
        )

    def mean_until(self, until: int) -> float | None:
        """Return the mean of the accuracy until the given cycle (inclusive)"""
        if until < 0:
            raise ValueError("`until` must be non-negative")

        if until >= len(self.entries):
            return None

        @functools.cache
        def helper(until: int):
            return float(
                np.mean([entry.accuracy for entry in self.entries[: (until + 1)]])
            )

        return helper(until)

    def means(self) -> list[float]:
        output: list[float] = []

        for i in range(len(self.entries)):
            mean = self.mean_until(i)
            assert mean is not None
            output.append(mean)

        return output

    def mean_drift(self, within: int) -> tuple[float, float] | None:
        """Get the minimum and maximum mean value within the final n cycles.

        Returns None if there isn't enough data
        """
        if len(self.entries) < within:
            return None

        means = self.means()

        bounded = means[-within:]

        return (min(bounded), max(bounded))

    def is_stable(self, within: int, threshold: float) -> bool:
        drift: tuple[float, float] | None = self.mean_drift(within)

        if drift is None:
            logger.debug(
                f"Not stable, not enough runs passed to compute mean drift ({within} required)"
            )
            return False

        drift_min, drift_max = drift

        drift_amount = drift_max - drift_min

        is_stable = drift_amount <= threshold
        if is_stable:
            logger.debug("Achieved stability")
        else:
            logger.debug(f"Not stable, {drift_amount}>{threshold}")

        return is_stable
