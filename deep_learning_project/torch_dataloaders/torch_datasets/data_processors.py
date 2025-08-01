"""
原始数据处理工具类。
约定为torch_datasets所使用的工具类。

大多数任务都会构建的工作流，约定:
    - 必要的，将原始数据转换为torch.Tensor。
    - 简单数据操作也独立构建processor，以便于可能的项目修改。

更好的实践:
    - 做的任务仅可能简易，主要为CPU进行的IO操作。
    - 仅包含简单数据操作，可以易于使用多线程方法。
    - 如果需要使用GPU的任务，预先进行加载和处理，而不是在dataset阶段进行。
"""

from __future__ import annotations

import torch
# from transformers import AutoProcessor
from pathlib import Path

from typing import TYPE_CHECKING, Any
# if TYPE_CHECKING:


class DataProcessor:
    """
    original data处理方法。

    以工具类进行构建，约定使用函数式编程。

    常见的工具方法:
        - 处理数据转换和数据类型转换。
        - 从指定路径加载数据。
        - 标签处理。
    """
    @staticmethod
    def process_data(
        data: Any,
    ) -> torch.Tensor:
        """
        一个方法的示例，需要根据具体任务进行修改。

        Args:
            data (Any): 各种形式都有可能的数据。

        Returns:
            torch.Tensor: 供模型计算的tensor。
        """

