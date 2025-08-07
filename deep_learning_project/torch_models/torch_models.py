"""
基于原生torch对于模型的定义。

基础依然是基于torch.nn.Module定义的模型。

约定:
    - 在这个位置的是整体模型，为实验需求需要易于变动。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


# TODO: 可以设计Interface以实现更好的通用性。
class NormalModel(nn.Module):
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        # 导入配置并分配。
        self.config = config
        self.backbone_config = config['backbone']
        self.choice = self.backbone_config['choice']
        self.backbone_model_config = self.backbone_config[self.choice]

        self.is_freeze = config['is_freeze']

        if self.is_freeze:
            self.freeze()

    def forward(
        self,
        inputs: torch.Tensor,
    ):
        outputs = self.backbone(inputs)
        return outputs

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

