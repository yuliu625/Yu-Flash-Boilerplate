"""
当前项目模型的整体及其变体。

是通用的基于nn.Module定义的模型。
"""

import torch
import torch.nn as nn


# TODO: 可以设计Interface以实现更好的通用性。
class CommonModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # 导入配置并分配。
        self.config = config
        self.backbone_config = config['backbone']
        self.choice = self.backbone_config['choice']
        self.backbone_model_config = self.backbone_config[self.choice]

        self.is_freeze = config['is_freeze']

        if self.is_freeze:
            self.freeze()

    def forward(self, inputs):
        outputs = self.backbone(inputs)
        return outputs

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    """在这里模拟输入，不需要真的使用dataloader。"""
    torch.Size([1, 19, 3, 224, 224])
    example_input_array = torch.rand(1, 5, 3, 224, 224)
    model = CommonModel({

    })
    outputs = model(example_input_array)
    print(outputs)
    print(outputs.shape)
