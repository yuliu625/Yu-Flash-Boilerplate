"""
原始的torch中的对于dataloader的定义。
"""

from __future__ import annotations

from deep_learning_project.torch_dataloaders.torch_datasets import DFDataset
from deep_learning_project.torch_dataloaders.collate_fn import collate_fn

from torch.utils.data import DataLoader

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class DataLoaderFactory:
    """
    以工厂模式设计。

    绑定的设计可以保证train_dataset和val_dataset是同一组。
    """
    def __init__(self, config: dict):
        """
        分配并传递dataset的设置。再设置dataloader。
        :param config:
            {
                dataset_global:
                    file:
                        control_file_dir_path: path
                        audio_dir_path: path
                    processor:
                        name: str
                        path: path

                train_dataset:
                    control_file:
                        path: ${torch_dataloaders.dataset_global.file.control_file_dir_path}/all_data.json
                    audio:
                        audio_dir_path: ${torch_dataloaders.dataset_global.file.audio_dir_path}
                    processor:
                        path: ${torch_dataloaders.dataset_global.processor.path}

                val_dataset:
                    control_file:
                        path: ${torch_dataloaders.dataset_global.file.control_file_dir_path}/all_data.json
                    audio:
                        audio_dir_path: ${torch_dataloaders.dataset_global.file.audio_dir_path}
                    processor:
                        path: ${torch_dataloaders.dataset_global.processor.path}

                dataloader_global:
                    batch_size: int
                    num_workers: int

                train_dataloader:
                    batch_size: ${torch_dataloaders.dataloader_global.batch_size}
                    num_workers: ${torch_dataloaders.dataloader_global.num_workers}
                    shuffle: True

                val_dataloader:
                    batch_size: ${torch_dataloaders.dataloader_global.batch_size}
                    num_workers: ${torch_dataloaders.dataloader_global.num_workers}
                    shuffle: False
            }
        """
        # 导入设置。
        self.config = config

        # 分配设置。
        self.train_dataloader_config = config['train_dataloader']
        self.val_dataloader_config = config['val_dataloader']

        # 传递dataset的设置。
        self.dataset_config = config['torch_datasets']
        self.train_dataset_config = self.dataset_config['train_dataset']
        self.val_dataset_config = self.dataset_config['val_dataset']

    def get_train_dataloader(self):
        # 设置dataset。
        train_dataset = DFDataset(self.train_dataset_config)
        # 设置dataloader。
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.train_dataloader_config['batch_size'],
            num_workers=self.train_dataloader_config['num_workers'],
            shuffle=self.train_dataloader_config['shuffle'],
            collate_fn=collate_fn,
        )
        return train_dataloader

    def get_val_dataloader(self):
        # 设置dataset。
        val_dataset = DFDataset(self.val_dataset_config)
        # 设置dataloader。
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.val_dataloader_config['batch_size'],
            num_workers=self.train_dataloader_config['num_workers'],
            shuffle=self.val_dataloader_config['shuffle'],
            collate_fn=collate_fn,
        )
        return val_dataloader


if __name__ == '__main__':
    """对dataloader的测试放在collate_fn中进行。"""
