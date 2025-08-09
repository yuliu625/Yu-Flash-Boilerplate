"""
基于torchmetrics的Metric自定义metric。

面临场景:
    - 具体应用: 构建更合适的评价指标。
"""

from __future__ import annotations

from torchmetrics import Metric

from typing import TYPE_CHECKING, Any
# if TYPE_CHECKING:


class CustomMetric(Metric):
    def update(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError

    def compute(self) -> Any:
        raise NotImplementedError

