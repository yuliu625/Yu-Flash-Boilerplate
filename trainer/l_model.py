"""
基于lightning的模型定义。

在我的设计中，model的定义依然是直接使用nn.Module来实现的，模型的设置也在其中实现。
仅在模型完整的实现之后，才在LightningModule中组合。
目的在于，低耦合，便于迁移。

这里仅设置干净的step相关内容。
"""

# from model import

import torch
import torch.nn as nn
import torch.optim as optim

import lightning as pl
import torchmetrics


class LightningModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config['model']

        # 设置模型
        self.model = MyModel(self.model_config)
        # 设置损失函数
        self.loss_fn = self.choose_loss_fn(config['loss_fn'])

        # 额外的设置
        # 保存模型配置的超参数。
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        """必要的，实现每一步的训练。但是batch_idx似乎用不到，是lightning会用的。"""

        # 获取数据
        labels = batch['labels']
        inputs = batch['datas']

        # 前向传播
        outputs = self.model(inputs)

        # 测试损失
        loss = self.loss_fn(outputs, labels)

        # 日志
        self.log('train_loss', loss)
        return loss  # 这里的返回是反向传播和优化器需要的。

    def configure_optimizers(self):
        """必要的，设置优化器。"""
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        """应该的，验证模型。"""

        # 获取数据
        labels = batch['labels']
        inputs = batch['datas']

        # 前向传播
        outputs = self.model(inputs)

        # 测试损失
        loss = self.loss_fn(outputs, labels)

        # 这里应该还有测试指标。

        # 日志
        self.log('val_loss', loss)

    def choose_loss_fn(self, choice: str):
        if choice == 'cross_entropy':
            return nn.CrossEntropyLoss()


if __name__ == '__main__':
    pass
