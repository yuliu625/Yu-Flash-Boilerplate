"""
基于原生torch，构建optimizer。
"""

from __future__ import annotations

import torch

from typing import TYPE_CHECKING, Literal
# if TYPE_CHECKING:


class OptimizerClassFactory:
    @staticmethod
    def create_optimizer_class(
        optimizer_name: Literal['Adam', 'SGD', 'RMSProp', 'RMSPropOptimizer'],
    ) -> type[torch.optim.Optimizer]:
        if optimizer_name == 'Adam':
            return OptimizerClassFactory.create_adam_optimizer_class()

    @staticmethod
    def create_optimizer(
        model: torch.nn.Module,
        optimizer_name: str,
    ) -> torch.optim.Optimizer:
        ...

    @staticmethod
    def create_adam_optimizer_class():
        return torch.optim.Adam

