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

from deep_learning_project.torch_models.torch_model_factory import TorchModelFactory

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from lightning import LightningModule


class LModelFactory:
    # ====暴露方法。====
    @staticmethod
    def create_lightning_model(
        torch_model_name: Literal[
            'normal',
        ],
        torch_model_config: dict,
        loss_fn_name: Literal[
            'cross_entropy', 'mse', 'binary_crossentropy'
        ],
        loss_fn_config: dict,
        optimizer_name: str,
        optimizer_config: dict,
        metrics_configs_list: list[dict],
    ) -> LightningModule:
        torch_model = TorchModelFactory.create_torch_model(
            torch_model_name=torch_model_name,
            torch_model_config=torch_model_config,
        )
        loss_fn = LossFnFactory.create_loss_fn(
            loss_fn_name=loss_fn_name,
            loss_fn_config=loss_fn_config,
        )
        optimizer_class = OptimizerClassFactory.create_optimizer_class(
            optimizer_name=optimizer_name,
        )
        metrics_list = [
            dict(
                metric_name=metric_config['metric_name'],
                metric_fn=MetricFactory.create_metric(
                    metric_name=metric_config['metric_name'],
                    metric_config=metric_config['metrics_kwargs'],
                )
            )
            for metric_config in metrics_configs_list
        ]
        l_model = LModel(
            torch_model=torch_model,
            loss_fn=loss_fn,
            optimizer_class=optimizer_class,
            optimizer_configs=optimizer_config,
            metrics_list=metrics_list,
        )
        return l_model

