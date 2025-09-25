from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
from matplotlib.axes import Axes
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from hamming_utils._stats import HammingStats

__all__ = ["Data"]

models: dict[str, nn.Module] = dict()
dataloaders: dict[str, DataLoader] = dict()


def get_dataloader(dataset_name: str) -> DataLoader:
    global dataloaders

    try:
        return dataloaders[dataset_name]
    except KeyError:
        print(f"Loading {dataset_name}")

    match dataset_name:
        case "cifar10":
            mean = (0.49139968, 0.48215827, 0.44653124)
            std = (0.24703233, 0.24348505, 0.26158768)
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            )
            dataset = torchvision.datasets.CIFAR10(
                root="./dataset_cache", train=False, download=True, transform=transform
            )
        case "cifar100":
            mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
            dataset = torchvision.datasets.CIFAR100(
                root="./dataset_cache", train=False, download=True, transform=transform
            )
        case _:
            raise ValueError(f"Unsupported dataset {dataset_name}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1000,
        shuffle=False,
    )
    dataloaders[dataset_name] = loader

    return loader


def get_model(model_name: str, dataset_name) -> nn.Module:
    global models

    name = f"{dataset_name}_{model_name}"

    try:
        return copy.deepcopy(models[name])
    except KeyError:
        print(f"Loading {name}")

    model = torch.hub.load("chenyaofo/pytorch-cifar-models", name, pretrained=True)  # type: ignore
    assert isinstance(model, nn.Module)

    models[name] = model
    return copy.deepcopy(model)


def build_accuracy_fn(dataloader: DataLoader, device: torch.device, model_name: str):
    def accuracy_fn(module: nn.Module, half: bool):
        nonlocal device
        nonlocal dataloader
        nonlocal model_name

        module = module.to(device)
        if half:
            module = module.half()

        module.eval()
        num_samples = torch.tensor(0).to(device)
        num_correct = torch.tensor(0).to(device)

        print(f"Running {model_name} on {device}")
        for data in dataloader:
            inputs, targets = data[0].to(device), data[1].to(device)
            assert isinstance(inputs, torch.Tensor)
            assert isinstance(targets, torch.Tensor)
            if half:
                inputs = inputs.half()
                targets = targets.half()

            outputs = module(inputs)

            # Convert logits to class indices
            outputs = outputs.argmax(dim=1)

            num_samples += targets.size(0)
            num_correct += (outputs == targets).sum()

        return (num_correct / num_samples * 100).item()

    return accuracy_fn


def data_file(path: str) -> Path:
    p = Path(path).expanduser()

    if p.is_dir():
        p = p.joinpath("data.json")

    if not p.exists():
        print(f"Creating a new data file at {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()

    return p


@dataclass
class MetaData:
    buffer_size: int | None
    dtype: str
    model: str
    dataset: str


@dataclass
class Data:
    entries: list[HammingStats]
    # how many bits to use for the protected buffer
    meta: MetaData

    @classmethod
    def load(cls, path: str, metadata: MetaData | None) -> Data:
        with open(data_file(path)) as f:
            str_data = f.read()

            if len(str_data) == 0:
                if metadata is not None:
                    return cls([], metadata)
                else:
                    raise RuntimeError(f"file {path} is empty")

            data = json.loads(str_data)

        if not isinstance(data, dict):
            raise ValueError(f"Expected a dictionary, got {type(data)}")

        entries = data["entries"]
        if not isinstance(entries, list):
            raise ValueError(f"Expected a list, got {type(entries)}")
        entries = [HammingStats.from_dict(x) for x in entries]

        buffer_size = data["buffer_size"]
        if not isinstance(buffer_size, int | None):
            raise ValueError(f"Expected in or None, got {type(buffer_size)}")

        dtype = data["dtype"]
        if not isinstance(dtype, str):
            raise ValueError(f"Expected a string, got {type(dtype)}")

        try:
            model = data["model"]
        except KeyError:
            model = "resnet20"

        if not isinstance(model, str):
            raise ValueError(f"Expected a string, got {type(model)}")

        try:
            dataset = data["dataset"]
        except KeyError:
            dataset = "cifar10"

        if not isinstance(dataset, str):
            raise ValueError(f"Expected a string, got {type(dataset)}")

        if metadata is not None and buffer_size != metadata.buffer_size:
            raise ValueError(
                f"Buffer size mismatch {buffer_size} vs {metadata.buffer_size}"
            )
        if metadata is not None and dtype != metadata.dtype:
            raise ValueError(f"Data type mismatch {dtype} vs {metadata.dtype}")

        if metadata is not None and model != metadata.model:
            raise ValueError(f"Model mismatch {model} vs {metadata.model}")

        if metadata is not None and dataset != metadata.dataset:
            raise ValueError(f"Dataset mismatch {dataset} vs {metadata.dataset}")

        return cls(entries, MetaData(buffer_size, dtype, model, dataset))

    def save(self, path: str) -> None:
        output = dict()
        output["entries"] = [x.to_dict() for x in self.entries]
        output["buffer_size"] = self.meta.buffer_size
        output["dtype"] = self.meta.dtype
        output["model"] = self.meta.model
        output["dataset"] = self.meta.dataset

        file = data_file(path)
        print(f"saving data to {file}")

        with open(file, "w") as f:
            json.dump(output, f)

    def record(
        self,
        bit_error_rate: float,
        half: bool,
        save_path: str,
        protected_buffer_size: int | None,
        *,
        autosave: bool = True,
        device: torch.device | None = None,
        summary: bool = True,
    ) -> None:
        if protected_buffer_size is not None:
            eval_fn = HammingStats.protected_eval(protected_buffer_size)
        else:
            eval_fn = HammingStats.unprotected_eval

        if device is None:
            device = torch.device("cpu")

        model = get_model(self.meta.model, self.meta.dataset)
        loader = get_dataloader(self.meta.dataset)

        stats = eval_fn(
            model,
            bit_error_rate,
            build_accuracy_fn(loader, device, self.meta.model),
            half,
        )
        if summary:
            stats.summary()
        self.entries.append(stats)

        if autosave:
            self.save(save_path)

    def record_n(
        self,
        n: int,
        bit_error_rate: float,
        half: bool,
        save_path: str,
        protected_buffer_size: int | None,
        *,
        autosave: bool | int = True,
        device: torch.device | None = None,
        summary: bool = True,
    ) -> None:
        if n < 1:
            raise ValueError("Expected at least 1 iteration")

        if isinstance(autosave, bool):
            if autosave:
                autosave = 1
            else:
                autosave = 0

        protected = protected_buffer_size is not None
        for i in range(n):
            print(
                f"recording {i + 1}/{n} {'un' if not protected else ''}protected inference"
            )
            save = autosave != 0 and (i + 1) % autosave == 0
            self.record(
                bit_error_rate,
                half,
                save_path,
                protected_buffer_size,
                autosave=save,
                device=device,
                summary=summary,
            )

        if autosave != 0:
            self.save(save_path)

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

        print(f"model: {self.meta.model}")
        print(f"dataset: {self.meta.dataset}")
        print(f"buffer size: {self.meta.buffer_size}")
        print(f"dtype: {self.meta.dtype}")
        print("Entries by Bit Error Rate:")
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

    def plot_accuracy(self, ax: Axes) -> None:
        bers = list(self.partition().items())
        bers.sort(key=lambda x: x[0])

        labels, data = list(zip(*bers))
        labels = [f"{label:.2e}" for label in labels]
        positions = [i for i in range(len(labels))]

        dtype = self.meta.dtype
        bsize = self.meta.buffer_size

        title = dtype + (
            " unprotected" if bsize is None else f" ECC {bsize} bit buffer"
        )
        title += f" - {self.entries[0].total_bits:.2e} total bits"

        ax.set_title(title)
        ax.violinplot(data, positions=positions, showmeans=True, showextrema=True)
        ax.set_xticks(ticks=positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Accuracy [%]")
        ax.set_xlabel("Bit Error Rate")
        ax.set_ylim(0, 100)
