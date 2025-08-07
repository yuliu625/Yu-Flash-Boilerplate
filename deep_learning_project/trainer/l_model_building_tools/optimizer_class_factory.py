"""
基于原生torch，构建optimizer。
"""

from __future__ import annotations

import torch

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class OptimizerClassFactory:
    @staticmethod
    def create_optimizer_class(
        optimizer_name: str,
    ) -> type[torch.optim.Optimizer]:
        ...

    @staticmethod
    def create_optimizer(
        model: torch.nn.Module,
        optimizer_name: str,
    ) -> torch.optim.Optimizer:
        ...

