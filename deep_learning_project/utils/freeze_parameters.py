"""
冻结模型指定部分的参数。

效果:
    - 模型的部分模块在前向传播过程中不会计算梯度，在反向传播过程中不会更新参数。
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch


def freeze_parameters(
    torch_module: torch.nn.Module,
) -> None:
    for param in torch_module.parameters():
        param.requires_grad = False

