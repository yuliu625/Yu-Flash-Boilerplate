"""
基于原生torch，构建loss_fn。

这个工厂方法构建的目的:
    - 完全以可序列化数据构建metric。
"""

from __future__ import annotations

# 同一文件夹下导入自定义的loss_fn。
# from .torch_loss_fns import (
#
# )

import torch

from typing import TYPE_CHECKING, Literal
# if TYPE_CHECKING:


class LossFnFactory:
    # ====暴露方法。====
    @staticmethod
    def create_loss_fn(
        loss_fn_name: Literal[
            'cross_entropy',
            'mse',
        ],
        loss_fn_config: dict,
    ) -> torch.nn.Module:
        if loss_fn_name == 'cross_entropy':
            return LossFnFactory.create_cross_entropy_loss_fn(loss_fn_config=loss_fn_config)

    """<!--分类任务-start-->"""

    @staticmethod
    def create_cross_entropy_loss_fn(
        loss_fn_config: dict,
    ) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss(**loss_fn_config)

    """<!--分类任务-end-->"""

    """<!--回归任务-start-->"""

    @staticmethod
    def create_mse_loss_fn(
        loss_fn_config: dict,
    ) -> torch.nn.Module:
        return torch.nn.MSELoss(**loss_fn_config)

    """<!--回归任务-end-->"""

    """<!--自定义-start-->"""

    """<!--自定义-end-->"""

