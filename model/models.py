"""
当前项目模型的整体及其变体。

是通用的基于nn.Module定义的模型。
"""

import torch
import torch.nn as nn


# TODO: 可以设计Interface以实现更好的通用性。
class MyModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        if self.config['is_freeze']:
            self.freeze()

    def forward(self, x):
        return x

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    pass
