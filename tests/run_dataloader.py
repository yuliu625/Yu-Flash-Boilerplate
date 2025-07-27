"""
测试dataloader的载入情况。

主要测试目的为：
    - collate_fn的工作情况。确保batch的正确。
"""

from .run_dataset import get_dataset

from deep_learning_project.torch_dataloaders.collate_fn import collate_fn

from torch.utils.data import DataLoader


def check_dataloader(config):
    common_dataset = get_dataset(config)

    dataloader = DataLoader(common_dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        break


if __name__ == '__main__':
    """这里指定config。之后会改进为从命令行或配置文件自动读取。"""
    check_dataloader({})
