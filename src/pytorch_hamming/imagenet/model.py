import copy
import enum
import logging

import timm
from torch import nn

_logger = logging.getLogger(__name__)

_root_module_cache: dict[str, nn.Module] = dict()


class Model(enum.Enum):
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
