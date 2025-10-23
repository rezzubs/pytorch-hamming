import enum
from dataclasses import dataclass

import torch
from torch import nn

from hamming_utils import (
    BaseSystem,
    MetaData,
)
from hamming_utils.systems import (
    CachedDataset,
    CachedModel,
)


class Dtype(enum.Enum):
    Float32 = enum.auto()
    Float16 = enum.auto()

    def to_torch(self) -> torch.dtype:
        match self:
            case Dtype.Float32:
                return torch.float32
            case Dtype.Float16:
                return torch.float16


def map_layer(module: nn.Module) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []

    assert isinstance(module.weight, torch.Tensor)
    tensors.append(module.weight)

    if module.bias is not None:
        assert isinstance(module.bias, torch.Tensor)
        tensors.append(module.bias)

    if not isinstance(module, nn.BatchNorm2d):
        return tensors

    if module.running_mean is not None:
        assert isinstance(module.bias, torch.Tensor)
        tensors.append(module.running_mean)

    if module.running_var is not None:
        assert isinstance(module.bias, torch.Tensor)
        tensors.append(module.running_var)

    return tensors


@dataclass
class System(BaseSystem):
    dataset: CachedDataset
    model: CachedModel
    dtype: Dtype
    device: torch.device

    def system_root_module(self) -> nn.Module:
        return self.model.root_module()

    def system_accuracy(
        self,
        root_module: nn.Module,
    ):
        root_module = root_module.to(self.device)

        root_module.eval()
        num_samples = torch.tensor(0).to(self.device)
        num_correct = torch.tensor(0).to(self.device)

        for data in self.dataset.loader():
            inputs, targets = data[0], data[1]
            assert isinstance(inputs, torch.Tensor)
            assert isinstance(targets, torch.Tensor)
            inputs = inputs.to(self.device).to(self.dtype.to_torch())
            targets = targets.to(self.device).to(self.dtype.to_torch())

            outputs = root_module(inputs)

            outputs = outputs.argmax(dim=1)

            num_samples += targets.size(0)
            num_correct += (outputs == targets).sum()

        return (num_correct / num_samples * 100).item()

    def system_data_tensors(self, root_module: nn.Module) -> list[torch.Tensor]:
        tensors = map_layer(root_module)

        for child in root_module.children():
            child_tensors = self.system_data_tensors(child)
            tensors.extend(child_tensors)

        return tensors

    def system_metadata(self) -> MetaData:
        return {
            "dtype": self.dtype.name,
            "model": self.model.kind.name,
            "dataset": self.dataset.name,
        }
