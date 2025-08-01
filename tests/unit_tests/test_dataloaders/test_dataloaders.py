"""
测试dataloader是否能正常加载数据。

测试dataloader和测试collate_fn是一体的。

结合测试dataset的情况，可以确定collate_fn是否按照预期工作。
"""

from __future__ import annotations

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


# """在这里测试dataloader及其collate_fn的情况。"""
# from torch_datasets import DFDataset
# from torch.utils.data import DataLoader
#
# # 这里可以直接使用dataset包中已经测试通过的dataset和其config。
# common_dataset = DFDataset({
#     'control_file': {'path': r''},
#     'file': {'file_dir_path': r''},
#     'processor': {'path': r''},
# })
#
# # 只要输出一次，batch_size大于1，并且有填充。
# dataloader = DataLoader(common_dataset, batch_size=2, collate_fn=collate_fn)
# for batch in dataloader:
#     print(batch)
#     break

