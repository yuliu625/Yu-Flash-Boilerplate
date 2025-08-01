"""
dataloader的重要参数collate_fn。

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

from typing import TYPE_CHECKING, Callable, Literal
# if TYPE_CHECKING:


class CollateFnFactory:
    # ====暴露方法。====
    @staticmethod
    def create_collate_fn(
        collate_fn_name: Literal['default'],
    ) -> Callable[..., dict[str, torch.Tensor]]:
        if collate_fn_name == 'default':
            return CollateFnFactory.create_default_collate_fn()

    # ====示例方法。根据具体任务进行修改和扩展。====
    @staticmethod
    def create_default_collate_fn() -> Callable[..., dict[str, torch.Tensor]]:
        def collate_fn(batch) -> dict[str, torch.Tensor]:
            # 以list形式读取。
            target_list = [data['target'] for data in batch]
            data_list = [data['data'] for data in batch]

            # 组合。
            targets = torch.stack(target_list)
            datas = pad_sequence(data_list, batch_first=True)

            # 以dict返回。
            return {
                'targets': targets,
                'datas': datas,
            }
        return collate_fn

