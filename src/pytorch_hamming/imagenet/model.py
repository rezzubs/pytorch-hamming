import copy
import enum
import logging
from typing import cast

import timm
from timm.data.config import (
    resolve_data_config,  # pyright: ignore[reportUnknownVariableType]
)
from timm.data.transforms_factory import create_transform
from torch import nn
from torchvision import transforms  # pyright: ignore[reportMissingTypeStubs]

from pytorch_hamming.imagenet.dataset import Transform

_logger = logging.getLogger(__name__)

_root_module_cache: dict[str, nn.Module] = dict()


class Model(enum.Enum):
    DeitTiny = "deit_tiny_patch16_224"
    SwinTiny = "swin_tiny_patch4_window7_224"
    VitBase = "vit_base_patch16_224"
    VitTiny = "vit_tiny_patch16_224"

    def load_root_module(self) -> nn.Module:
        _logger.info(f"Loading model {self.value}")
        return timm.create_model(self.value, pretrained=True)

    def get_root_module(self) -> nn.Module:
        """Get a copy of a cached root module."""
        root_module = _root_module_cache.get(self.value, None)

        if root_module is not None:
            _logger.debug("Using cached module")
            return copy.deepcopy(root_module)

        root_module = self.load_root_module()
        _root_module_cache[self.value] = root_module

        return copy.deepcopy(root_module)

    def get_transform(self) -> Transform:
        """Get the proper preprocessing transform for this model."""

        model = self.load_root_module()

        pretrained_cfg = model.pretrained_cfg
        config = resolve_data_config(pretrained_cfg=pretrained_cfg)  # pyright: ignore[reportUnknownVariableType]
        assert isinstance(config, object)

        if not isinstance(config, dict):
            raise TypeError(f"Expected config to be a dict, got {type(config)}")

        transform = create_transform(**config)  # pyright: ignore[reportUnknownArgumentType]

        if not isinstance(transform, transforms.Compose):
            raise TypeError(
                f"Expected transform to be a Compose, got {type(transform)}"
            )

        return cast(Transform, transform)
