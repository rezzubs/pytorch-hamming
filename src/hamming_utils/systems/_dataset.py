from __future__ import annotations

from collections.abc import Generator, Iterator
import enum
import logging
import typing

from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms


logger = logging.getLogger(__name__)

Dataset = datasets.CIFAR10 | datasets.CIFAR100

ON_DISK_CACHE = "./dataset_cache"


CACHE: dict[CachedDataset, Dataset] = dict()


class CachedDataset(enum.Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"

    def load_data(self) -> Dataset:
        logger.info(f"Loading dataset `{self.name}`")
        match self:
            case CachedDataset.CIFAR10:
                mean = (0.49139968, 0.48215827, 0.44653124)
                std = (0.24703233, 0.24348505, 0.26158768)
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize(mean, std)]
                )
                return datasets.CIFAR10(
                    root=ON_DISK_CACHE,
                    train=False,
                    download=True,
                    transform=transform,
                )
            case CachedDataset.CIFAR100:
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

                return datasets.CIFAR100(
                    root=ON_DISK_CACHE,
                    train=False,
                    download=True,
                    transform=transform,
                )

    def dataset(self) -> Dataset:
        dataset = CACHE.get(self)

        if dataset is None:
            logger.debug(f"Dataset {self.name} not yet cached")
            dataset = self.load_data()
            CACHE[self] = dataset
        else:
            logger.debug(f"Got dataset {self.name} from cache")

        return dataset

    def loader(
        self, batch_size: int = 1000
    ) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
        dataloader = typing.cast(
            DataLoader[list[torch.Tensor]],
            DataLoader(
                self.dataset(),
                batch_size=batch_size,
                shuffle=False,
            ),
        )

        # NOTE: The following is necessary because pytorch doesn't provide a
        # type safe API for `DataLoader`.
        iterator = typing.cast(Iterator[list[torch.Tensor]], iter(dataloader))
        for batch in iterator:
            assert isinstance(batch, list)
            assert len(batch) == 2
            assert all(isinstance(x, torch.Tensor) for x in batch)

            yield (batch[0], batch[1])

        return None
