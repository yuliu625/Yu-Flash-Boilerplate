"""
构建评测指标的方法。
"""

from __future__ import annotations

import torchmetrics

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torchmetrics import Metric


class MetricFactory:
    @staticmethod
    def create_metric(
        metric_name: str,
        metric_kwargs: dict,
    ) -> Metric:
        ...

    @staticmethod
    def create_accuracy_metric(
        metric_kwargs: dict,
    ) -> Metric:
        return torchmetrics.Accuracy(**metric_kwargs)

