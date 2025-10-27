from __future__ import annotations

import abc
from typing import Generic, TypeVar

import torch

from .tensor_ops import (
    tensor_list_num_bits,
)


T = TypeVar("T")


class BaseSystem(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def system_data(self) -> T:
        """Return the data that will be used as input to other functions expecting `data`.

        The data should be the thing component which is affected by fault
        injection into data tensors. In the case of a DNN it would most likely
        be the root module but it can really be anything.
        """

    @abc.abstractmethod
    def system_accuracy(self, data: T) -> float:
        """Get the accuracy of the system for the given `data`"""

    @abc.abstractmethod
    def system_data_tensors(self, data: T) -> list[torch.Tensor]:
        """Get references to tensors that make up this `data`"""

    def system_metadata(self) -> dict[str, str]:
        """Return metadata about the system.

        This will be used to uniquely identify the system.
        """
        return dict()

    def system_total_num_bits(self) -> int:
        return tensor_list_num_bits(self.system_data_tensors(self.system_data()))
