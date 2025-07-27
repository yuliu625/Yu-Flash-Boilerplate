"""
模型中使用到的模块的定义。会在model包中使用。

全部都是nn.Module的子类。
会有独立的相关的测试。
"""

# __all__ = []


# 一般情况下，仅对外暴露各种结构的model。
from .torch_models import CommonModel
