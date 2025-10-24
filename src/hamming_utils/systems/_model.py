from __future__ import annotations

import enum
from dataclasses import dataclass

import torch
from torch import nn
import copy
import logging

logger = logging.getLogger(__name__)

ROOT_MODULE_CACHE: dict[CachedModel.Kind, nn.Module] = dict()


@dataclass
class CachedModel:
    kind: CachedModel.Kind

    class Kind(enum.Enum):
        ResNet20 = "resnet20"
        VGG11 = "vgg11_bn"

    def root_module(self) -> nn.Module:
        model = ROOT_MODULE_CACHE.get(self.kind)

        if model is None:
            model = torch.hub.load(  # pyright: ignore[reportUnknownMemberType]
                "chenyaofo/pytorch-cifar-models", self.kind.value, pretrained=True
            )
            assert isinstance(model, nn.Module)
            ROOT_MODULE_CACHE[self.kind] = model

        return copy.deepcopy(model)
