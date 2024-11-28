from torch.utils.data import Dataset

import pandas as pd

from pathlib import Path
from omegaconf import OmegaConf, DictConfig


class MyDataset(Dataset):
    """
    torch中实现dataset的基础。

    优化流程，复杂的处理由相关的类服务来实现。
    """
    def __init__(self, config: DictConfig):
        """
        我的做法：
            - 由一个控制文件管理，导入使用pandas。
            - 根据路径读取文件和处理由另一个类来实现。
        """

    def __len__(self):
        pass
        # return len()

    def __getitem__(self, index):
        pass


if __name__ == '__main__':
    pass
