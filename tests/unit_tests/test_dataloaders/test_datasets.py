"""
测试dataset是否能正常加载数据。
"""

from __future__ import annotations
import pytest

from deep_learning_project.torch_dataloaders.torch_datasets.torch_dataset_factory import TorchDatasetFactory

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class TestDatasets:
    def test_train_dataset(self, train_dataset_config):
        train_dataset = TorchDatasetFactory.create_train_dataset(
            dataset_config=train_dataset_config,
        )
        print(len(train_dataset))
        print(train_dataset[0])
        print(train_dataset[1])

    def test_val_dataset(self, val_dataset_config):
        val_dataset = TorchDatasetFactory.create_train_dataset(
            dataset_config=val_dataset_config,
        )
        print(len(val_dataset))
        print(val_dataset[0])
        print(val_dataset[1])

