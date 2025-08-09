"""
基于原生torch的各种场景的dataset的定义。

dataset的职责:
    - 加载数据集。
    - 处理数据，进行必要的转换。
        - 基础的，需要输出是torch.Tensor。

2种实现方法:
    - 预加载: 加载并管理全部的数据。
        - 延迟低，数据处理方便。
        - 需要很大的内容。通常在数据预测任务和文本任务使用。
    - 懒加载: 只加载必要的控制文件，大规模的实际数据会在训练阶段被加载。
        - 可以使用超级大的数据集。通常用于图像、视频等任务。
        - 需要额外的文件管理方法。
        - 数据处理需要提前在data_processing中做好，需要有多版本管理方法。
"""

from __future__ import annotations

# 从同一文件夹导入数据处理工具方法。
# from .data_processors import (
#
# )

import torch
import pandas as pd
from pathlib import Path

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class DFDataset(torch.utils.data.Dataset):
    """
    torch中实现dataset的基础。

    此处的实现:
        - 读取数据由pandas完成。
        - 创建的是可随机读写的dataset。

    更好的实践:
        - 轻数据处理，必要的数据处理由分离的processor工具类实现。
        - 考虑统一的train_dataset和val_dataset加载方法。
    """
    def __init__(
        self,
        df: pd.DataFrame,
    ):
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        # 加载 (这个方法就是df本身。)

        # 处理

        # 组合并返回
        return dict(
            # 'target':
            # 'data':
        )


class ControlDataset(torch.utils.data.Dataset):
    """
    复杂一点的情况，数据不能完全读入内存，由控制文件进行管理。

    约定控制文件的内容:
        - 标签或目标值。
        - 读取实际数据的路径或方法。
    """
    def __init__(
        self,
        control_df: pd.DataFrame,
    ):
        self.control_df = control_df

    def __len__(self) -> int:
        """
        控制文件的长度即实际数据的长度。。

        Returns:
            int: 数据的数量。
        """
        return len(self.control_df)

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        """
        约定的较好实践:
            - 这个方法进行的操作:
                - 根据控制文件加载数据。
                - 使用处理工具类处理数据。
                - 组合为dict返回。
            - 以dict形式返回，约定:
                - key1: 'target' 分类任务或回归任务的目标。
                - key2: 'data' 输入模型的数据。
                    如果是多模态数据，可以增加kv，但是需要在这里已经处理好。
        """
        # 加载

        # 处理

        # 组合并返回
        return dict(
            # 'target':
            # 'data':
        )

    def process_data(
        self,
        data,
    ) -> torch.Tensor:
        """
        可以用一个工具类来实现，处理完需要是tensor。
        """

