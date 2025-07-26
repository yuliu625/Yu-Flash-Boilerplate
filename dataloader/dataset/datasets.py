"""
基于原生pytorch的各种场景的dataset的定义。
"""

from __future__ import annotations

from dataloader.dataset.processor import Processor

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class CommonDataset(Dataset):
    """
    torch中实现dataset的基础。

    数据处理以processor来实现。
    需要在一开始同时考虑统一的train_dataset和val_dataset的实现。
    """
    def __init__(self, config: dict):
        """
        我的做法：
            - 由一个控制文件管理，导入使用pandas。
            - 根据路径读取文件和处理由另一个类来实现。
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
                    path: ${dataloader.dataset_global.file.control_file_dir_path}/all_data.json
                audio:
                    audio_dir_path: ${dataloader.dataset_global.file.audio_dir_path}
                processor:
                    path: ${dataloader.dataset_global.processor.path}
        }
        """
        self.config = config
        self.control_df = pd.read_json(Path(self.config['control_file']['path']))

        # self.processor = Processor(self.config['processor'])

    def __len__(self):
        """统一的控制管理，这里直接返回控制文件的长度。"""
        return len(self.control_df)

    def __getitem__(self, index) -> dict:
        """
        我的做法：
            - 分为3步。
                - 根据控制文件加载数据。
                - 使用处理类处理数据。
                - 组合为dict返回。
            - 以dict形式返回，有很大好处。
        :param index:
        :return: dict
        """
        # 加载

        # 处理

        # 组合并返回
        return {

        }

    def process_data(self, data) -> torch.Tensor:
        """可以用一个工具类来实现，处理完需要是tensor。"""


if __name__ == '__main__':
    """实例化一个dataset，查看数据输出的情况。可先查看path的映射情况。"""
    common_dataset = CommonDataset({
        'control_file': {'path': r''},
        'file': {'file_dir_path': r''},
        'processor': {'path': r''},
    })

    print(len(common_dataset))
    print(common_dataset[0])
    print(common_dataset[1])
