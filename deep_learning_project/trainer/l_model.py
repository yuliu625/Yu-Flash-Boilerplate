"""
基于lightning的Module。

无法避免的，这个类需要进行具体的修改，也可能需要多种实现。

LightningModule的特点:
    - 模型与训练绑定: 除了基础的前向传播，还绑定了反向传播相关的方法。
    - 自动管理: 如果在trainer中同时以LightningDataModule进行训练，大量的操作能够自动被管理。

我约定的实现:
    - 最高层封装: 仅在最高层组合所有的模块，或者仅在最高层封装原有的模型。目的在于:
        - 兼容性: 原始模型依然是通用的基于torch.nn.Module定义的。
        - 低耦合: 视pl.LightningModule为构建训练的一部分。
        - 效率: 开发对lightning依赖程度低，训练和监视高于原生torch。
"""

from __future__ import annotations

import torch
import lightning as pl

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torchmetrics


class LModel(pl.LightningModule):
    """
    基于lightning构建模型。
    """
    def __init__(
        self,
        torch_model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer_class: type[torch.optim.Optimizer],
        optimizer_configs: dict,
        metrics_dict: dict[str, torchmetrics.Metric],
    ):
        super().__init__()
        # 设置模型
        self.torch_model = torch_model
        # 设置损失函数
        self.loss_fn = loss_fn
        # 设置优化器
        self.optimizer_class = optimizer_class
        self.optimizer_configs = optimizer_configs
        # 设置评估函数
        self.metrics_dict = metrics_dict
        # self.accuracy_fn = torchmetrics.Accuracy(task='multiclass', num_classes=8)
        # self.precision_fn = torchmetrics.Precision(task='multiclass', num_classes=8)
        # self.recall_fn = torchmetrics.Recall(task='multiclass', num_classes=8)
        # self.f1_score_fn = torchmetrics.F1Score(task='multiclass', average='weighted', num_classes=8)

        # 额外的设置。
        # 保存模型配置的超参数。由于低耦合的设计模式，这个方法并没有被使用。
        # self.save_hyperparameters()
        # 测试模型的输入，会在summary table中展现。
        # self.example_input_array = torch.rand(4, 2000)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        一些情况下，需要实现forward来避免报错。
        """
        outputs: torch.Tensor = self.torch_model(inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        """
        必要的，实现每一步的训练。但是batch_idx似乎用不到，是lightning会用的。
        """
        # 获取数据
        targets: torch.Tensor = batch['targets']
        inputs: torch.Tensor = batch['datas']
        # 前向传播
        outputs: torch.Tensor = self.torch_model(inputs)
        # 计算训练损失
        loss: torch.Tensor = self.loss_fn(outputs, targets)
        # 日志
        self.log('train_loss', loss)
        return loss  # 这里的返回是反向传播和优化器需要的。

    def configure_optimizers(self):
        """
        必要的，设置优化器。
        """
        optimizer = self.optimizer_class(
            params=self.torch_model.parameters(),
            **self.optimizer_config,
        )
        return optimizer

    def validation_step(self, batch, batch_idx):
        """
        应该的，验证模型。
        """
        # 获取数据
        targets: torch.Tensor = batch['targets']
        inputs: torch.Tensor = batch['datas']
        # 前向传播
        outputs: torch.Tensor = self.torch_model(inputs)
        # 测试损失
        loss: torch.Tensor = self.loss_fn(outputs, targets)
        self.log('val_loss', loss)
        # 测试指标。
        metrics_log = {}
        for metric_name, metric in self.metrics_dict.items():
            # 进行评测计算
            metrics_value = metric(outputs, targets)
            # 日志记录。
            self.log(f'{metric_name}', metrics_value)
        return metrics_log

