"""
基于原生torch对于模型的定义。

基础依然是基于torch.nn.Module定义的模型，分离设计以实现与具体训练器的低耦合。

约定:
    - 在这个位置的是整体模型，为实验需求需要易于变动。
"""

from __future__ import annotations

# from .torch_modules import (
#
# )

import torch

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class DemoModel(torch.nn.Module):
    def __init__(
        self,
        torch_model_config: dict,
    ):
        super().__init__()
        # 导入配置并分配。
        self.config = torch_model_config
        self.backbone_config = torch_model_config['backbone']
        self.choice = self.backbone_config['choice']
        self.backbone_model_config = self.backbone_config[self.choice]

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        outputs: torch.Tensor = self.backbone(inputs)
        return outputs

