from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from hamming_utils._stats import HammingStats

__all__ = ["Data"]

dataloader: DataLoader | None = None
resnet: nn.Module | None = None


def get_dataloader() -> DataLoader:
    global dataloader

    if dataloader is not None:
        return dataloader

    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    testset = torchvision.datasets.CIFAR10(
        root="./dataset_cache", train=False, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1000,
        shuffle=False,
    )

    return dataloader


def get_resnet() -> nn.Module:
    global resnet

    if resnet is not None:
        return copy.deepcopy(resnet)

    resnet = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
    )  # type: ignore
    assert isinstance(resnet, nn.Module)

    return copy.deepcopy(resnet)


def data_file(path: str) -> Path:
    p = Path(path).expanduser()

    if p.is_dir():
        p = p.joinpath("data.json")

    if not p.exists():
        print(f"Creating a new data file at {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()

    return p


def evaluate_resnet(device: torch.device):
    def inner(model: nn.Module, half: bool):
        global dataloader
        nonlocal device

        model = model.to(device)
        if half:
            model = model.half()

        model.eval()
        num_samples = torch.tensor(0).to(device)
        num_correct = torch.tensor(0).to(device)

        print(f"Running resnet on {device}")
        for data in get_dataloader():
            inputs, targets = data[0].to(device), data[1].to(device)
            assert isinstance(inputs, torch.Tensor)
            assert isinstance(targets, torch.Tensor)
            if half:
                inputs = inputs.half()
                targets = targets.half()

            outputs = model(inputs)

            # Convert logits to class indices
            outputs = outputs.argmax(dim=1)

            num_samples += targets.size(0)
            num_correct += (outputs == targets).sum()

        return (num_correct / num_samples * 100).item()

    return inner


@dataclass
class Data:
    entries: list[HammingStats]
    autosave_path: str | None

    @classmethod
    def load(cls, path: str) -> Data:
        with open(data_file(path)) as f:
            str_data = f.read()
            if len(str_data) == 0:
                data = []
            else:
                data = json.loads(str_data)

        if not isinstance(data, list):
            raise ValueError(f"Expected a list, got {type(data)}")

        mapped = [HammingStats.from_dict(x) for x in data]

        return cls(mapped, path)

    def save(self, path: str) -> None:
        serializable = [x.to_dict() for x in self.entries]

        file = data_file(path)
        print(f"saving data to {file}")

        with open(file, "w") as f:
            json.dump(serializable, f)

    def record(
        self,
        bit_error_rate: float,
        protected: bool,
        half: bool,
        *,
        autosave: bool = True,
        device: torch.device | None = None,
        summary: bool = True,
        data_buffer_size: int = 64,
    ) -> None:
        if protected:
            eval_fn = HammingStats.eval
        else:
            eval_fn = HammingStats.eval_noprotect

        if device is None:
            device = torch.device("cpu")

        stats = eval_fn(
            get_resnet(),
            bit_error_rate,
            evaluate_resnet(device),
            data_buffer_size,
            half,
        )
        if summary:
            stats.summary()
        self.entries.append(stats)

        if autosave:
            self.save(self.autosave_path or "./")

    def record_n(
        self,
        n: int,
        bit_error_rate: float,
        protected: bool,
        half: bool,
        *,
        autosave: bool | int = True,
        device: torch.device | None = None,
        summary: bool = False,
        data_buffer_size: int = 64,
    ) -> None:
        if n < 1:
            raise ValueError("Expected at least 1 iteration")

        if isinstance(autosave, bool):
            if autosave:
                autosave = 1
            else:
                autosave = 0

        for i in range(n):
            print(
                f"recording {i + 1}/{n} {'un' if not protected else ''}protected inference"
            )
            save = autosave != 0 and (i + 1) % autosave == 0
            self.record(
                bit_error_rate,
                protected,
                half,
                autosave=save,
                device=device,
                summary=summary,
                data_buffer_size=data_buffer_size,
            )

        if autosave != 0:
            self.save(self.autosave_path or "./")

    def partition(self) -> dict[float, list[float]]:
        """Group the accuracy metrics by bit error rate."""

        bers: dict[float, list[float]] = dict()

        for entry in self.entries:
            ber = entry.bit_error_rate()
            ber_group = bers.get(ber, [])

            ber_group.append(entry.accuracy)

            bers[ber] = ber_group

        return bers

    def overview(self) -> None:
        """Print an overview of the current dataset."""

        bers = list(self.partition().items())
        bers.sort(key=lambda x: x[0])

        print("Bit Error Rate")
        for i, (ber, entries) in enumerate(bers):
            if i != len(bers) - 1:
                prefix0 = "├── "
                prefix1 = "│   "
            else:
                prefix0 = "└── "
                prefix1 = "    "

            print(prefix0 + f"{ber:.3}")
            print(prefix1 + f"├── runs: {len(entries)}")
            print(prefix1 + f"├── mean: {np.mean(entries):.3}")
            print(prefix1 + f"└── std: {np.std(entries):.3}")
