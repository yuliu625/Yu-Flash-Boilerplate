"""
构建dataset的工厂。
"""

from __future__ import annotations

from torch.utils.data import Dataset

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class DatasetFactory:
    @staticmethod
    def create_dataset(

    ) -> Dataset:
        ...

