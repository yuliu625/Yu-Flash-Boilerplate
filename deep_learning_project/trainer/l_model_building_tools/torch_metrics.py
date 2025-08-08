"""
基于torchmetrics的Metric自定义metric。

面临场景:
    - 具体应用: 构建更合适的评价指标。
"""

from __future__ import annotations

from torchmetrics import Metric

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class CustomMetric(Metric):
    ...

