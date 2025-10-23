from __future__ import annotations

import enum
import logging

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

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

    def loader(self, batch_size=1000) -> DataLoader:
        return DataLoader(
            self.dataset(),
            batch_size=batch_size,
            shuffle=False,
        )
