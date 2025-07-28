"""
基于原生pytorch的各种场景的dataset的定义。
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class DFDataset(Dataset):
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
        """
        我的做法：
            - 由一个控制文件管理，导入使用pandas。
            - 根据路径读取文件和处理由另一个类来实现。
        """
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> dict:
        """
        我的做法：
            - 分为3步。
                - 根据控制文件加载数据。
                - 使用处理类处理数据。
                - 组合为dict返回。
            - 以dict形式返回，有很大好处。
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
        """可以用一个工具类来实现，处理完需要是tensor。"""


class ControlDataset(Dataset):
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

    def __getitem__(self, index) -> dict:
        ...

