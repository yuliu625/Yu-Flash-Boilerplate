"""
测试dataloader是否能正常加载数据。

注意:
    - 测试dataloader和测试collate_fn是一体的。

目的:
    - 查看batch的结果。
    - 结合测试dataset的情况，可以确定collate_fn是否按照预期工作。
"""

from __future__ import annotations
import pytest

from deep_learning_project.torch_dataloaders.torch_datasets.torch_dataset_factory import TorchDatasetFactory
from deep_learning_project.torch_dataloaders.torch_dataloader_factory import TorchDataLoaderFactory

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class TestDataLoaders:
    def test_train_dataloader(self, train_dataset_config, train_dataloader_config):
        train_dataset = TorchDatasetFactory.create_train_dataset(
            dataset_config=train_dataset_config,
        )
        train_dataloader = TorchDataLoaderFactory.create_train_dataloader(
            train_dataset=train_dataset,
            train_dataloader_config=train_dataloader_config,
        )
        for batch in train_dataloader:
            print(batch)
            break

    def test_val_dataloader(self, val_dataset_config, val_dataloader_config):
        val_dataset = TorchDatasetFactory.create_val_dataset(
            dataset_config=val_dataset_config,
        )
        val_dataloader = TorchDataLoaderFactory.create_val_dataloader(
            val_dataset=val_dataset,
            val_dataloader_config=val_dataloader_config,
        )
        for batch in val_dataloader:
            print(batch)
            break

