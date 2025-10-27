from __future__ import annotations

import abc

import torch
from torch import nn

from .tensor_ops import (
    tensor_list_num_bits,
)


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

    def system_metadata(self) -> dict[str, str]:
        """Return metadata about the system.

        This will be used to uniquely identify the system.
        """
        return dict()

    def system_total_num_bits(self) -> int:
        return tensor_list_num_bits(self.system_data_tensors(self.system_root_module()))
