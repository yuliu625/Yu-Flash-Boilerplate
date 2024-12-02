"""
dataloader的collate_fn很重要。

经过collate_fn：
    - 原本键的名称会改变，我的方式是设置为复数。
    - batch中不同的length会被padding。
我的做法，会以dict返回。
"""

import torch
from torch.nn.utils.rnn import pad_sequence


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
    pass
