"""
构建dataset的工厂。

以完全序列化的方法获取dataset对象，从而极大简化后续所有工程和测试。
"""

from __future__ import annotations

# 同一文件夹下导入定义的各种dataset。
# from .torch_datasets import

from torch.utils.data import Dataset

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class DatasetFactory:
    @staticmethod
    def create_dataset(

    ) -> Dataset:
        ...

    @staticmethod
    def create_train_dataset(

    ) -> Dataset:
        ...

    @staticmethod
    def create_validate_dataset(

    ) -> Dataset:
        ...

    @staticmethod
    def create_test_dataset(

    ) -> Dataset:
        ...

    @staticmethod
    def create_predict_dataset(

    ) -> Dataset:
        ...

