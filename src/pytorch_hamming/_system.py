from __future__ import annotations

import abc

import torch
import copy

from .tensor_ops import (
    tensor_list_fault_injection,
    tensor_list_num_bits,
)


class BaseSystem[T](abc.ABC):
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
        """Get references to tensors that make up this `data`

        These are the tensors that are going to be used for comparing two systems of the same type.

        By default the same tensors are used as the target of fault injection.
        If this is not desired, `system_fault_injector` can be overridden.
        """

    def system_inject_n_faults(self, data: T, n: int):
        """Inject `n` faults uniformly into the system `data`."""
        tensors = self.system_data_tensors(data)
        tensor_list_fault_injection(tensors, n)

    def system_metadata(self) -> dict[str, str]:
        """Return metadata about the system.

        This will be used to uniquely identify the system.
        """
        return dict()

    def system_clone_data(self, data: T) -> T:
        """Clone the data.

        The default implementation uses `deepcopy` but this behavior can be
        overriden.
        """
        return copy.deepcopy(data)

    def system_total_num_bits(self) -> int:
        return tensor_list_num_bits(self.system_data_tensors(self.system_data()))
