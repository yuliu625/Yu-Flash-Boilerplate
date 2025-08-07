"""
构建dataset的工厂。

以完全序列化的方法获取dataset对象，从而极大简化后续所有工程和测试。

注意，这个方法直接和该项目中自定义的data-module对应，方法可以不真正实现，但是需要保留。
"""

from __future__ import annotations

# 同一文件夹下导入定义的各种dataset。
# from .torch_datasets import

from torch.utils.data import Dataset

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class DatasetFactory:
    """
    各种dataset的构建方法。
    """
    # ====暴露方法。需要实现的方法。====
    @staticmethod
    def create_train_dataset(
        data_path,
    ) -> Dataset:
        ...

    # ====暴露方法。需要实现的方法。====
    @staticmethod
    def create_validate_dataset(
        data_path,
    ) -> Dataset:
        ...

    # ====暴露方法。需要实现的方法。====
    @staticmethod
    def create_test_dataset(
        data_path,
    ) -> Dataset:
        ...

    # ====暴露方法。需要实现的方法。====
    @staticmethod
    def create_predict_dataset(
        data_path,
    ) -> Dataset:
        ...

