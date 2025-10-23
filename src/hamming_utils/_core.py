from __future__ import annotations

import abc
import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel
from torch import nn

from hamming_utils.tensor_ops import (
    tensor_list_compare_bitwise,
    tensor_list_fault_injection,
)


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

    def record_entry(self, system: BaseSystem):
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
            tensor_list_fault_injection(data_tensors, self.num_faults)

        self.entries.append(
            Data.Entry(
                accuracy=system.system_accuracy(root),
                faulty_parameters=tensor_list_compare_bitwise(
                    original_tensors, data_tensors
                ),
            )
        )

    def save(self, data_path: str):
        with open(Path(data_path).expanduser(), "w") as f:
            f.write(self.model_dump_json())

    def load(self, data_path: str):
        with open(Path(data_path).expanduser(), "r") as f:
            content = f.read()
            Data.model_validate_json(content)
