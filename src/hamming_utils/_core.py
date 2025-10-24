from __future__ import annotations

import abc
import copy
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel
from torch import nn

from .tensor_ops import (
    tensor_list_compare_bitwise,
    tensor_list_fault_injection,
)

logger = logging.getLogger(__name__)


class MetaDataError(Exception):
    """The metadata didn't match"""


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


class Data(BaseModel):
    num_faults: int
    num_bits: int
    metadata: MetaData
    entries: list[Data.Entry]

    @dataclass
    class Entry(BaseModel):
        accuracy: float
        faulty_parameters: list[int]

    def record_entry(self, system: BaseSystem) -> None:
        """Record a new data entry for the given `system`"""

        logger.debug("Recording new data entry")

        if system.system_metadata() != self.metadata:
            raise MetaDataError(
                "Data has different metadata than the given system:\n"
                f"data: {self.metadata}\n"
                f"system: {system.system_metadata()}"
            )

        root = copy.deepcopy(system.system_root_module())

        data_tensors = system.system_data_tensors(root)
        original_tensors = copy.deepcopy(data_tensors)

        if self.num_faults > 0:
            logger.debug("Running fault injection")
            tensor_list_fault_injection(data_tensors, self.num_faults)
        else:
            logger.debug("Skipping fault injection")

        self.entries.append(
            Data.Entry(
                accuracy=system.system_accuracy(root),
                faulty_parameters=tensor_list_compare_bitwise(
                    original_tensors, data_tensors
                ),
            )
        )

    def save(self, data_path: str) -> None:
        """Save the data to `data_path`."""

        path = Path(data_path).expanduser()

        if path.exists():
            logger.info(f'Saving data to "{path}"')
        else:
            logger.info(f'Saving data to a new file at "{path}"')

        with open(path, "w") as f:
            f.write(self.model_dump_json())

    @classmethod
    def load_or_create(
        cls, data_path: str, *, num_faults: int, num_bits: int, metadata: MetaData
    ) -> Data:
        """Load existing data from disk or create a new instance if it doesn't exist."""

        path = Path(data_path).expanduser()

        if not path.exists():
            logger.warning(
                f'Didn\'t find existing data at "{path}", creating a new instance'
            )
            return cls(
                num_faults=num_faults, num_bits=num_bits, metadata=metadata, entries=[]
            )

        if not path.is_dir():
            logger.warning(f'The path "{path}" is not a file, saving to it will fail"')

        logger.info('Loading existing data from "{path}"')

        with open(path, "r") as f:
            content = f.read()
            return Data.model_validate_json(content)
