# from dataloader import
from trainer.l_model import LightningModel

import torch
import lightning as pl

from omegaconf import OmegaConf, DictConfig


class LightningTrainer:
    """
    封装了lightning trainer的流程。
    """
    def __init__(self, config: DictConfig) -> None:
        """
        :param config: 这是一个总的配置文件。
        """
        self.config = config

        self.train_dataloader = None
        self.val_dataloader = None

        self.model = self.build_model()
        self.trainer = self.build_trainer()

    def build_model(self):
        """构建模型。"""
        model = LightningModel(self.config)
        # 断点续训判断
        if self.config.resume:
            model.load_from_checkpoint(self.config.resume)
        return model

    def build_trainer(self):
        trainer = pl.Trainer(default_root_dir=self.config['default_root_dir'],)
        return trainer

    def build_callbacks(self):
        pass

    def train(self):
        """进行训练"""
        self.trainer.fit(
            model=self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader
        )




if __name__ == '__main__':
    pass
