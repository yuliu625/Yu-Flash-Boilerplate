"""
基于原生torch的dataloader的所相关的实现。

使用场景:
    - 原生torch: 直接在兼容torch的框架种使用dataloader。
    - lightning: 实际上并不直接使用dataloader，而是定义lightning的data-module。
"""

from .torch_datasets import TorchDatasetFactory
from .torch_dataloader_factory import TorchDataLoaderFactory

