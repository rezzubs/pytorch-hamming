from __future__ import annotations

import enum
import torch


class Dtype(enum.Enum):
    Float32 = enum.auto()
    Float16 = enum.auto()

    def to_torch(self) -> torch.dtype:
        match self:
            case Dtype.Float32:
                return torch.float32
            case Dtype.Float16:
                return torch.float16

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> Dtype:
        match dtype:
            case torch.float32:
                return Dtype.Float32
            case torch.float16:
                return Dtype.Float16
            case other:
                raise ValueError(f"Unsupported pytorch data type {other}")
