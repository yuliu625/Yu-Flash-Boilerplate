"""
dataloader会使用的collate_fn。

collate_fn的功能应该具有:
    - 以list进入的独立的值转换为一组。
    - 不同长度的序列使用相关padding方法处理为相同长度。

约定:
    - 区别名称: 进入数据以单数命名，collate_fn处理后的数据以复数命名。
    - 可使用dict。复杂情况，使用dict进行控制。
"""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


def collate_fn(batch):
    # 以list形式读取。
    label_list = [data['label'] for data in batch]
    data_list = [data['data'] for data in batch]

    # 组合。
    labels = torch.stack(label_list)
    datas = pad_sequence(data_list, batch_first=True)

    # 以dict返回。
    return {
        'labels': labels,
        'datas': datas,
    }


if __name__ == '__main__':
    """在这里测试dataloader及其collate_fn的情况。"""
    from dataset import CommonDataset
    from torch.utils.data import DataLoader

    # 这里可以直接使用dataset包中已经测试通过的dataset和其config。
    common_dataset = CommonDataset({
        'control_file': {'path': r''},
        'file': {'file_dir_path': r''},
        'processor': {'path': r''},
    })

    # 只要输出一次，batch_size大于1，并且有填充。
    dataloader = DataLoader(common_dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        break
