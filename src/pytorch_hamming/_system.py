from __future__ import annotations

import abc

import torch

from .tensor_ops import (
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
    def system_data_tensors_cmp(self, data: T) -> list[torch.Tensor]:
        """Get references to tensors that make up this `data`

        These are the tensors that are going to be used for comparing two systems of the same type.
        """

    def system_data_tensors_fi(self, data: T) -> list[torch.Tensor]:
        """Get references to tensors that make up this `data`.

        These are the tensors that are going to be used for fault injection. The
        default implementation uses the same value as `system_data_tensors_cmp`
        but it can be overriden to use a different representation for fault
        injection.
        """
        return self.system_data_tensors_cmp(data)

    def system_metadata(self) -> dict[str, str]:
        """Return metadata about the system.

        This will be used to uniquely identify the system.
        """
        return dict()

    def system_total_num_bits(self) -> int:
        return tensor_list_num_bits(self.system_data_tensors_cmp(self.system_data()))
