"""
基于torch.nn.Module自定义的loss_fn。

面临场景:
    - 复杂目标: 构建更合适的引导方法。
"""

from __future__ import annotations

import torch

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class CustomLoss(torch.nn.Module):
    ...

