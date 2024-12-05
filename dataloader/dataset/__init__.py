"""
数据集的定义，封装了数据的加载。会在dataloader包中直接使用。
"""

# __all__ = []


# 一般情况下，仅对外暴露定义的相关dataset，供dataloader使用。
from .datasets import CommonDataset
