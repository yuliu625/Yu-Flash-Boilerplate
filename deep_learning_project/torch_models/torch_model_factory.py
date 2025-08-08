"""
生成基于torch定义的model的工厂方法。
"""

from __future__ import annotations

from .torch_models import (
    DemoModel,
)

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    import torch


class TorchModelFactory:
    @staticmethod
    def create_torch_model(
        torch_model_name: Literal['demo'],
        torch_model_config: dict,
    ) -> torch.nn.Module:
        if torch_model_name == 'demo':
            return TorchModelFactory.create_demo_torch_model(torch_model_config=torch_model_config)

    @staticmethod
    def create_demo_torch_model(
        torch_model_config: dict,
    ) -> torch.nn.Module:
        return DemoModel(**torch_model_config)

