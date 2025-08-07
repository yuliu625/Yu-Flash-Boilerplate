"""
基于原生torch，构建loss_fn。
"""

from __future__ import annotations

import torch

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class LossFnFactory:
    @staticmethod
    def create_loss_fn(
        loss_fn_name: str,
        loss_fn_kwargs: dict,
    ) -> torch.nn.Module:
        ...

    @staticmethod
    def create_cross_entropy_loss_fn(
        loss_fn_kwargs: dict,
    ) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss(
            **loss_fn_kwargs,
        )

