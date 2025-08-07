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

from deep_learning_project.torch_models import NormalModel

import torch
import torch.nn as nn
import torch.optim as optim
import lightning as pl
import torchmetrics

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor


class LModel(pl.LightningModule):
    """
    lightning中本身的做法。

    我根据lightning的设计，对配置文件做了适配。
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer_class: type[optim.Optimizer],
        optimizer_kwargs: dict,
        metrics,
    ):
        super().__init__()

        # 设置模型
        self.model = model
        # 设置损失函数
        self.loss_fn = self.choose_loss_fn(self.criterion_config['loss_fn']['name'])
        # 设置优化器
        # self.optimizer = optim.Adam(self.torch_models.parameters(), lr=self.model_config['lr'])
        # 设置评估函数
        self.accuracy_fn = torchmetrics.Accuracy(task='multiclass', num_classes=8)
        self.precision_fn = torchmetrics.Precision(task='multiclass', num_classes=8)
        self.recall_fn = torchmetrics.Recall(task='multiclass', num_classes=8)
        self.f1_score_fn = torchmetrics.F1Score(task='multiclass', average='weighted', num_classes=8)

        # 额外的设置。
        # 保存模型配置的超参数。
        self.save_hyperparameters()
        # 测试模型的输入，会在summary table中展现。
        self.example_input_array = torch.rand(4, 2000)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        一些情况下，需要实现forward来避免报错。
        """
        outputs: torch.Tensor = self.model(inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        """
        必要的，实现每一步的训练。但是batch_idx似乎用不到，是lightning会用的。
        """
        # 获取数据
        targets: torch.Tensor = batch['targets']
        inputs: torch.Tensor = batch['datas']
        # 前向传播
        outputs: torch.Tensor = self.model(inputs)
        # 计算训练损失
        loss: torch.Tensor = self.loss_fn(outputs, targets)
        # 日志
        self.log('train_loss', loss)
        return loss  # 这里的返回是反向传播和优化器需要的。

    def configure_optimizers(self):
        """
        必要的，设置优化器。
        """
        optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=self.config.lr,
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
        outputs: torch.Tensor = self.model(inputs)
        # 测试损失
        loss: torch.Tensor = self.loss_fn(outputs, targets)
        # 这里应该还有测试指标。
        accuracy: torch.Tensor = self.accuracy_fn(outputs, targets)
        precision: torch.Tensor = self.precision_fn(outputs, targets)
        recall: torch.Tensor = self.recall_fn(outputs, targets)
        f1_score: torch.Tensor = self.f1_score_fn(outputs, targets)

        # 日志
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_f1_score', f1_score)
        # 也可以用self.log_dict({})。
        return {
            'val_loss': loss,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1_score': f1_score,
        }

    def choose_loss_fn(self, choice: str):
        if choice == 'cross_entropy':
            return nn.CrossEntropyLoss()

    def choose_metrics(self, choice: str):
        # TODO: 这个要同时设置多个metrics该如何实现呢？直接self中添加，validation中如何知道？
        # 似乎完全需要构建一个类才能实现，但是好像没有必要。要实现就需要动态添加属性，并同时在validation_step进行同步。
        # 但是，metrics方法自然是越多越好，并且不怎么改动的。
        if choice == 'accuracy':
            return torchmetrics.Accuracy()

