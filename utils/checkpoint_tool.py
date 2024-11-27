"""
这里是强化的对于checkpoint的保存和加载。
默认处理model、optimizer、scheduler。

设计了没有scheduler的情况，但是按照我的trainer的设计，scheduler是一定有的。

后来这里主要的方法被lightning中的ModelCheckpoint替代了，不过我依然会再加入一些特定情况的功能。
"""

import torch

from pathlib import Path


def save_checkpoint(path_to_save: str, model: torch.nn.Module, optimizer: torch.optim, scheduler=None):
    """保存需要继续训练的相关状态。"""
    # 自动处理路径文件夹情况。
    file_path = Path(path_to_save)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # 对于scheduler的额外处理。
    if scheduler is not None:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, file_path)
    else:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, file_path)


def load_checkpoint(path_to_checkpoint: str, model: torch.nn.Module, optimizer: torch.optim, scheduler=None):
    """加载需要继续训练的相关状态。"""
    checkpoint = torch.load(path_to_checkpoint)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # 对于scheduler的额外处理。
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
        return model, optimizer, scheduler
    else:
        return model, optimizer


if __name__ == '__main__':
    pass
