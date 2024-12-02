"""
lightning中的DataModule。

这个类可以代替torch中原本的dataloader相关的设置，在分布式环境下有很大好处。
但是为了通用性，我依然更多的使用原本的dataloader来定义数据的加载和处理。
"""

import torch

import lightning as pl
from torch.utils.data import DataLoader


class LDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader()

    def val_dataloader(self):
        return DataLoader()

    def test_dataloader(self):
        return DataLoader()

    def predict_dataloader(self):
        return DataLoader()

    def teardown(self, stage: str) -> None:
        pass


if __name__ == '__main__':
    pass
