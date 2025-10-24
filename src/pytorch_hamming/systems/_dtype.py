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
