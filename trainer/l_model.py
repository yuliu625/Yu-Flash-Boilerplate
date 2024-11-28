# from model import

import torch
import torch.optim as optim
import lightning as pl


class LightningModel(pl.LightningModule):
    def __init__(self, model, loss_fn, config):
        super().__init__()
        self.config = config
        self.model = model
        self.loss_fn = loss_fn

        # 额外的设置
        # 保存模型配置的超参数。
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        """必要的，实现每一步的训练。但是batch_idx似乎用不到，是lightning会用的。"""

        # 前向传播
        x, y = batch
        y_hat = self.model(x)

        # 计算损失
        loss = self.loss_fn(y_hat, y)

        # 日志
        self.log('train_loss', loss)
        return loss  # 这里的返回是反向传播和优化器需要的。

    def configure_optimizers(self):
        """必要的，设置优化器。"""
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        """应该的，验证模型。"""

        # 前向传播
        x, y = batch
        y_hat = self.model(x)

        # 测试损失
        loss = self.loss_fn(y_hat, y)

        # 这里应该还有测试指标。

        # 日志
        self.log('val_loss', loss)


if __name__ == '__main__':
    pass
