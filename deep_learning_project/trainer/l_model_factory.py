"""
构建用于训练的LModel的工厂。

职责:
    - 反序列化: 实现可以由序列化数据构建LModel对象。
    - 模型架构控制: 提供多种模型的变体。
    - 训练方法控制: 提供各种训练的设置。
"""

from __future__ import annotations

from .l_model_building_tools import (
    LossFnFactory,
    OptimizerClassFactory,
    MetricFactory,
)
from .l_model import LModel

# 需要从torch_models模块导入基于torch定义的模型。
# 可改进的，使用factory-pattern封装获取torch-model的方法。
# 暂时无法避免的，需要不断修改这个导入内容。
from deep_learning_project.torch_models.torch_models import NormalModel

import torch

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class LModelFactory:
    @staticmethod
    def create_default_model(
        loss_fn_name: str,
        loss_fn_kwargs: dict,
        optimizer_name: str,
        optimizer_config: dict,
        metrics_configs_list: list[dict]
    ):
        loss_fn = LossFnFactory.create_loss_fn(
            loss_fn_name=loss_fn_name,
            loss_fn_kwargs=loss_fn_kwargs,
        )
        optimizer_class = OptimizerClassFactory.create_optimizer_class(
            optimizer_name=optimizer_name,
        )
        metrics_dict = {
            metric_config['metric_name']: MetricFactory.create_metric(
                metric_name=metric_config['metric_name'],
                metric_kwargs=metric_config['metrics_kwargs'],
            )
            for metric_config in metrics_configs_list
        }
        l_model = LModel(
            torch_model=NormalModel({}),
            loss_fn=loss_fn,
            optimizer_class=optimizer_class,
            optimizer_configs=optimizer_config,
            metrics_list=metrics_dict,
        )
        return l_model

