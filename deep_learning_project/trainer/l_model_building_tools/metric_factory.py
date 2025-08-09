"""
基于torchmetrics构建评测指标的方法。

在我的deeplearning项目中，所有的metric都由torchmetrics库来实现。好处在于:
    - 更加高效: 以torch.Tensor在并行计算硬件上进行。
    - 分布式环境支持: 在分布式环境下不会犯错。
    - 可扩展: 需要的情况下，可基于Metric的Interface自定义metric。

这个工厂方法构建的目的:
    - 完全以可序列化数据构建metric。
"""

from __future__ import annotations

# 同一文件夹下导入自定义的loss_fn。
# from .torchmetrics_metrics import (
#
# )

import torchmetrics

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from torchmetrics import Metric


class MetricFactory:
    # ====暴露方法。====
    @staticmethod
    def create_metric(
        metric_name: Literal[
            'accuracy', 'precision', 'recall', 'f1score',
            'mse', 'mae', 'rmse', 'r2score',
        ],
        metric_config: dict,
    ) -> Metric:
        # 分类任务。
        if metric_name == 'accuracy':
            return MetricFactory.create_accuracy_metric(metric_config=metric_config)
        elif metric_name == 'precision':
            return MetricFactory.create_precision_metric(metric_config=metric_config)
        elif metric_name == 'recall':
            return MetricFactory.create_recall_metric(metric_config=metric_config)
        elif metric_name == 'f1score':
            return MetricFactory.create_f1score_metric(metric_config=metric_config)
        # 回归任务。
        elif metric_name == 'mse':
            return MetricFactory.create_mse_metric(metric_config=metric_config)
        elif metric_name == 'mae':
            return MetricFactory.create_mae_metric(metric_config=metric_config)
        elif metric_name == 'rmse':
            return MetricFactory.create_rmse_metric(metric_config=metric_config)
        elif metric_name == 'r2score':
            return MetricFactory.create_r2score_metric(metric_config=metric_config)

    """<!--分类任务-start-->"""

    @staticmethod
    def create_accuracy_metric(
        metric_config: dict,
    ) -> Metric:
        return torchmetrics.Accuracy(**metric_config)

    @staticmethod
    def create_precision_metric(
        metric_config: dict,
    ) -> Metric:
        return torchmetrics.Precision(**metric_config)

    @staticmethod
    def create_recall_metric(
        metric_config: dict,
    ) -> Metric:
        return torchmetrics.Recall(**metric_config)

    @staticmethod
    def create_f1score_metric(
        metric_config: dict,
    ) -> Metric:
        return torchmetrics.F1Score(**metric_config)

    """<!--分类任务-end-->"""

    """<!--回归任务-start-->"""

    @staticmethod
    def create_mse_metric(
        metric_config: dict,
    ) -> Metric:
        return torchmetrics.MeanSquaredError(**metric_config)

    @staticmethod
    def create_mae_metric(
        metric_config: dict,
    ) -> Metric:
        return torchmetrics.MeanAbsoluteError(**metric_config)

    @staticmethod
    def create_rmse_metric(
        metric_config: dict,
    ) -> Metric:
        return torchmetrics.NormalizedRootMeanSquaredError(**metric_config)

    @staticmethod
    def create_r2score_metric(
        metric_config: dict,
    ) -> Metric:
        return torchmetrics.R2Score(**metric_config)

    """<!--回归任务-end-->"""

    """<!--自定义-start-->"""

    """<!--自定义-end-->"""

