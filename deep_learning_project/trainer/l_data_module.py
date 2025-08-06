"""
lightning提供的DataModule。

对于常见任务，实现dataset_factory和dataloader_factory之后，一般不需要大规模修改这个文件。

优越性:
    - 自动管理数据，自行控制CPU和GPU的memory。
    - 分布式环境支持。
    - 极易迁移。

约定:
    - LDataModule的实例化完全由序列化数据实现。
    - 额外构建DatasetFactory和DataloaderFactory，并不在LDataModule中指定过多方法。
"""

from __future__ import annotations

from deep_learning_project.torch_dataloaders.torch_datasets.torch_dataset_factory import DatasetFactory
from deep_learning_project.torch_dataloaders.torch_dataloader_factory import DataLoaderFactory

import lightning as pl

from typing import TYPE_CHECKING, Literal
# if TYPE_CHECKING:


class LDataModule(pl.LightningDataModule):
    """
    lightning文档中的必要实现。

    约定:
        - 这个类的实例化完全由序列化数据实现。

    面对场景:
        - 单机多卡环境，不复杂的分布式环境。
        - 数据存储在本地。

    目前已有的实现:
        - 常见阶段的方法。

    扩展需要的需要的实现:
        - 实现各种额外的hook。
    """
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        # 这里仅放简单的设置和超参数。
        self.config = config

    # 在我约定的实现中，并不会使用这个方法。
    # 这个方法的意义是，在分布式环境下，限制相关方法仅由主节点执行。
    # 仅执行一次的操作可以由额外逻辑实现，而不是在DataModule中构建。
    # def prepare_data(self):
    #     ...

    def setup(
        self,
        stage=None,
    ) -> None:
        """
        L.Trainer会自动调用的方法。

        实现:
            - 实例化各种dataset。
            - 提供该类中dataloader构建需要的属性。

        额外说明:
            实际上，我很不喜欢这样写这个方法，原因在于:
                - stage输入是有限的，应指定为 stage: Literal['fit', 'validate', 'test', 'predict'],
                - stage传入None会复杂相关操作。
                - dataset在init方法之外构建。
                - 多个if判断和不明原因的None判断。
            但是，最终这样实现的原因是:
                - 这是lightning的惯例方法，我满足约定优于配置的编程风格。
                - 这个方法由L.Trainer控制，人为干预较少。
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = DatasetFactory.create_train_dataset()
            self.val_dataset = DatasetFactory.create_validate_dataset()
        if stage == 'validate' or stage is None:
            self.val_dataset = DatasetFactory.create_validate_dataset()
        if stage == 'test' or stage is None:
            self.test_dataset = DatasetFactory.create_test_dataset()
        if stage == 'predict' or stage is None:
            self.predict_dataset = DatasetFactory.create_predict_dataset()

    def train_dataloader(self):
        return DataLoaderFactory.create_train_dataloader(
            train_dataset=self.train_dataset,
        )

    def val_dataloader(self):
        return DataLoaderFactory.create_val_dataloader(
            val_dataset=self.val_dataset,
        )

    def test_dataloader(self):
        return DataLoaderFactory.create_test_dataloader(
            test_dataset=self.test_dataset,
        )

    def predict_dataloader(self):
        return DataLoaderFactory.create_predict_dataloader(
            predict_dataset=self.predict_dataset,
        )

    def teardown(
        self,
        stage: Literal['fit', 'validate', 'test', 'predict'],
    ) -> None:
        if stage == 'fit':
            self.train_dataset = None
            self.val_dataset = None
        if stage == 'validate':
            self.val_dataset = None
        if stage == 'test':
            self.test_dataset = None
        if stage == 'predict':
            self.predict_dataset = None

