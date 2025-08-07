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

import lightning as pl

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    import torchmetrics


class LModel(pl.LightningModule):
    """
    基于lightningModule构建的模型。

    这是个足够好的样例，但必要的情况下可以进行修改。

    没有进行规范项目的类型标注的原因是:
        - lightning项目的约定大于配置惯例。
        - 存在为复杂项目修改的空间。
    """
    def __init__(
        self,
        torch_model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer_class: type[torch.optim.Optimizer],
        optimizer_config: dict,
        metrics_list: list[dict[str, str | torchmetrics.Metric]],
    ):
        super().__init__()
        # 设置模型
        self.torch_model = torch_model
        # 设置损失函数
        self.loss_fn = loss_fn
        # 设置优化器
        self.optimizer_class = optimizer_class
        self.optimizer_config = optimizer_config
        # 设置评估函数
        self.metrics_list = metrics_list

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
        继承torch.nn.Module的forward方法。

        实现这个方法，以对外表现得和torch-model一致。
        """
        outputs: torch.Tensor = self.torch_model(inputs)
        return outputs

    # ====必要的实现。====
    def training_step(self, batch, batch_idx):
        """
        必要的训练步。

        实现:
            - 前向传播。
            - 计算损失。

        可为复杂项目进行的修改:
            - 自定义优化器计算过程。

        注意:
            - batch_idx参数在该方法中不显式使用，但lightning的自动运行会使用，不可删除。
        """
        # 获取数据。
        targets: torch.Tensor = batch['targets']
        inputs: torch.Tensor = batch['datas']
        # 前向传播。
        outputs: torch.Tensor = self.torch_model(inputs)
        # 计算训练损失。
        train_loss: torch.Tensor = self.loss_fn(outputs, targets)
        # 日志。
        self.log(
            name='train_loss',
            value=train_loss,
        )
        return train_loss  # 这里的返回是反向传播和优化器需要的。

    # ====必要的实现。====
    def configure_optimizers(self):
        """
        必要的，设定优化器。

        可为复杂项目进行的修改:
            - 多优化器。适应场景为:
                - 多任务。
                - 多阶段。
                - 复合模型。
        """
        optimizer = self.optimizer_class(
            params=self.torch_model.parameters(),
            **self.optimizer_config,
        )
        return optimizer

    # ====必要的实现。====
    def validation_step(self, batch, batch_idx):
        """
        应该的，验证模型步。

        对于模型的结果进行评测。
        """
        # 获取数据。
        targets: torch.Tensor = batch['targets']
        inputs: torch.Tensor = batch['datas']
        # 前向传播。
        outputs: torch.Tensor = self.torch_model(inputs)
        # 测试损失。
        val_loss: torch.Tensor = self.loss_fn(outputs, targets)
        self.log(
            name='val_loss',
            value=val_loss,
        )
        # 测试指标。
        metrics_log = {}
        for metric_dict in self.metrics_list:
            # 进行评测计算。
            metrics_value = metric_dict['metric_fn'](outputs, targets)
            # 日志记录。
            self.log(
                name=f'{metric_dict['metric_name']}',
                value=metrics_value,
            )
        return val_loss

