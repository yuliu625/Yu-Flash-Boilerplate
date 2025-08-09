"""
构建dataset的工厂。

以完全序列化的方法获取dataset对象，从而极大简化后续所有工程和测试。

注意:
    - 这个方法直接和该项目中自定义的data-module对应，方法可以不真正实现，但是需要保留。
        一种兼容实现为: 使用val_dataset相关的配置假装test和predict。
"""

from __future__ import annotations

# 同一文件夹下导入定义的各种dataset。
from .torch_datasets import (
    DFDataset,
    ControlDataset,
)

import torch
from pathlib import Path

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class TorchDatasetFactory:
    """
    获取各种dataset对象的方法。
    """
    # ====暴露方法。需要实现的方法。====
    @staticmethod
    def create_train_dataset(
        dataset_config: dict,
    ) -> torch.utils.data.Dataset:
        ...

    # ====暴露方法。需要实现的方法。====
    @staticmethod
    def create_val_dataset(
        dataset_config: dict,
    ) -> torch.utils.data.Dataset:
        ...

    # ====暴露方法。需要实现的方法。====
    @staticmethod
    def create_test_dataset(
        dataset_config: dict,
    ) -> torch.utils.data.Dataset:
        ...

    # ====暴露方法。需要实现的方法。====
    @staticmethod
    def create_predict_dataset(
        dataset_config: dict,
    ) -> torch.utils.data.Dataset:
        ...

