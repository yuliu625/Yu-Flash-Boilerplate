"""
所有模型都会有处理流，将原本的数据转换为tensor。

这里可能会涉及很多工具，这些都是在cpu上运行的。
如果使用transformers中的工具，需要进行导入。
"""

from transformers import AutoProcessor

import torch

from pathlib import Path


class Processor:
    """
    original data的处理方法。

    这里以工具的方式将processor与原本的dataset类分离。
    """
    def __init__(self, config: dict):
        self.config = config

    def __call__(self, data) -> torch.Tensor:
        """主要的方法，输入原本的数据，输出可供模型计算的tensor。"""


if __name__ == '__main__':
    """实例化一个processor，输入一个测试数据，查看处理的结果。"""
    processor = Processor({'path': r''})
    result = processor('data')

    print(result)
    print(type(result))
    print(result.dtype)
    print(result.shape)
