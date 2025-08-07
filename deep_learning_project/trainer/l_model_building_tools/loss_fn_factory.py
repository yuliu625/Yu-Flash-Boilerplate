"""
构建损失函数的工厂。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable
# if TYPE_CHECKING:


class LossFnFactory:
    @staticmethod
    def create_fn():
        ...

