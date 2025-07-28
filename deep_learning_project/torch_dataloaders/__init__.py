"""
dataloader的定义，输出为batch的tensor。最终输入模型的数据格式。
"""

# __all__ = []


# 暴露用工厂模式封装的dataloader获取类。或许用lightning的DataModule会更好？
from .torch_dataloaders import DataLoaderFactory
