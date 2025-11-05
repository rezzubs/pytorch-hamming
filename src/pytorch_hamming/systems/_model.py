from __future__ import annotations

import enum

from ._dataset import CachedDataset
import torch
from torch import nn
import copy
import logging

logger = logging.getLogger(__name__)

ROOT_MODULE_CACHE: dict[CachedModel, nn.Module] = dict()


class CachedModel(enum.Enum):
    ResNet20 = "resnet20"
    VGG11 = "vgg11_bn"

    def root_module(self, dataset: CachedDataset) -> nn.Module:
        model = ROOT_MODULE_CACHE.get(self)

        if model is None:
            model = torch.hub.load(  # pyright: ignore[reportUnknownMemberType]
                "chenyaofo/pytorch-cifar-models",
                f"{dataset.kind.value}_{self.value}",
                pretrained=True,
            )
            assert isinstance(model, nn.Module)
            ROOT_MODULE_CACHE[self] = model

        return copy.deepcopy(model)
