"""
lightning提供的DataModule。

优越性:
    - 自动管理数据，自行控制CPU和GPU的memory。
    - 分布式环境支持。

这个类可以代替torch中原本的dataloader相关的设置，在分布式环境下有很大好处。
为了通用性，我依然更多的使用原本的dataloader来定义数据的加载和处理。
而完全是自己的工程，我会使用这个模块。
"""

from __future__ import annotations

from deep_learning_project.torch_dataloaders.torch_datasets import DFDataset
from deep_learning_project.torch_dataloaders.collate_fn import collate_fn

import lightning as pl
from torch.utils.data import DataLoader

from typing import Literal
# if TYPE_CHECKING:


class LDataModule(pl.LightningDataModule):
    """
    就是原本lighting中DataModule该进行的设置。

    绝大多数情况下，我只会定义train和val。
    """
    def __init__(self, config: dict):
        # 这里仅放简单的设置和超参数。
        super().__init__()
        self.config = config

    # def prepare_data(self):
    #     """
    #     这个方法可以注释掉，因为一般用不到。
    #     我仅在文档中看到使用这个方法下载数据集。
    #     """

    def setup(self, stage: Literal['fit', 'validate', 'test', 'predict'] = None) -> None:
        """
        在这里实现原本dataset的实例化。
        将各种dataset的动态加载到对象属性上。虽然更好的工程代码不应该这么做，但是这里这样写很方便。
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = DFDataset(self.config)
            self.val_dataset = DFDataset(self.config)
        if stage == 'validate' or stage is None:
            self.val_dataset = DFDataset(self.config)
        # if stage == 'test' or stage is None:
        #     self.test_dataset = CommonDataset(self.config)
        # if stage == 'predict' or stage is None:
        #     self.predict_dataset = CommonDataset(self.config)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=collate_fn,
            batch_size=self.config['batch_size'],
            shuffle=self.config['shuffle'],
            num_workers=self.config['num_workers'],
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            collate_fn=collate_fn,
            batch_size=self.config['batch_size'],
            shuffle=self.config['shuffle'],
            num_workers=self.config['num_workers'],
        )

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset,)

    # def predict_dataloader(self):
    #     return DataLoader(self.predict_dataset,)

    def teardown(self, stage: Literal['fit', 'validate', 'test', 'predict'] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = None
            self.val_dataset = None
        if stage == 'validate' or stage is None:
            self.val_dataset = None

