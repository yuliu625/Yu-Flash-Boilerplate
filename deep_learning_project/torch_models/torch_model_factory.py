"""
生成基于torch定义的model的工厂方法。
"""

from __future__ import annotations

from .torch_models import (
    NormalModel,
)

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    import torch


class TorchModelFactory:
    @staticmethod
    def create_torch_model(
        torch_model_name: Literal['normal'],
        torch_model_config: dict,
    ) -> torch.nn.Module:
        if torch_model_name == 'normal':
            return TorchModelFactory.create_normal_torch_model(torch_model_config=torch_model_config)

    @staticmethod
    def create_normal_torch_model(
        torch_model_config: dict,
    ) -> torch.nn.Module:
        return NormalModel(**torch_model_config)

