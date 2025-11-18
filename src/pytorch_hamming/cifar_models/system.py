import logging
import typing
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from typing_extensions import override

from pytorch_hamming.cifar_models.dataset import CachedDataset
from pytorch_hamming.cifar_models.model import CachedModel
from pytorch_hamming.dtype import DnnDtype
from pytorch_hamming.system import BaseSystem

logger = logging.getLogger(__name__)


def _append_parameter(module: nn.Module, tensors: list[torch.Tensor], name: str):
    param = getattr(module, name, None)

    if param is None:
        return

    if not isinstance(param, torch.Tensor):
        logger.warning(f"Skipping parameter `{name}` because ({type(param)}!=Tensor)")  # pyright: ignore[reportAny]
        return

    tensors.append(param)


def map_layer(module: nn.Module) -> list[torch.Tensor]:
    """Extract the parameter tensors from a layer"""
    tensors: list[torch.Tensor] = []

    _append_parameter(module, tensors, "weight")
    _append_parameter(module, tensors, "bias")
    _append_parameter(module, tensors, "running_mean")
    _append_parameter(module, tensors, "running_var")

    return tensors


@dataclass
class System(BaseSystem[nn.Module]):
    dataset: CachedDataset
    model: CachedModel
    dtype: DnnDtype
    device: torch.device
    batch_size: int
    dataset_cache: Path | None

    @override
    def system_data(self) -> nn.Module:
        return self.model.root_module(self.dataset).to(self.dtype.to_torch())

    @override
    def system_accuracy(
        self,
        data: nn.Module,
    ) -> float:
        data = data.to(self.device)

        _ = data.eval()
        num_samples = torch.tensor(0).to(self.device)
        num_correct = torch.tensor(0).to(self.device)

        for data_unit in self.dataset.batches(
            self.batch_size,
            self.dtype.to_torch(),
            self.device,
        ):
            inputs, targets = data_unit[0], data_unit[1]
            inputs = inputs.to(self.device).to(self.dtype.to_torch())
            targets = targets.to(self.device).to(self.dtype.to_torch())

            outputs = typing.cast(torch.Tensor, data(inputs))
            assert isinstance(outputs, torch.Tensor)

            outputs = outputs.argmax(dim=1)

            num_samples += targets.size(0)
            num_correct += (outputs == targets).sum()

        return (num_correct / num_samples * 100).item()

    @override
    def system_data_tensors(self, data: nn.Module) -> list[torch.Tensor]:
        tensors = map_layer(data)

        for child in data.children():
            child_tensors = self.system_data_tensors(child)
            tensors.extend(child_tensors)

        return tensors

    @override
    def system_metadata(self) -> dict[str, str]:
        return {
            "dtype": self.dtype.name,
            "model": self.model.name,
            "dataset": self.dataset.kind.name,
        }
