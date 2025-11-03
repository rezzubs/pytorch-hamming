import abc
from typing import override
from dataclasses import dataclass
from pytorch_hamming.tensor_ops import tensor_list_fault_injection
import torch


class BaseFaultInjector(abc.ABC):
    @abc.abstractmethod
    def fault_injector_inject_n(self, faults_count: int) -> None: ...


@dataclass
class TensorListFaultInjector(BaseFaultInjector):
    tensors: list[torch.Tensor]

    @override
    def fault_injector_inject_n(self, faults_count: int) -> None:
        tensor_list_fault_injection(self.tensors, faults_count)
