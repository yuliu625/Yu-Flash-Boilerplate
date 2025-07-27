"""
封装了lightning的trainer。

这个boilerplate最重要的部分。
封装大量的重复配置项。几乎不用重写。
"""

from __future__ import annotations

from deep_learning_project.torch_dataloaders import DataLoaderFactory
from deep_learning_project.trainer.l_model import LightningModel
from deep_learning_project.trainer.l_callback import CallbackFactory
from deep_learning_project.trainer.l_logger import LoggerFactory

import lightning as pl
from omegaconf import OmegaConf

# if TYPE_CHECKING:


class LightningTrainer:
    """
    封装了lightning trainer的流程。
    """
    def __init__(self, config: dict) -> None:
        """
        :param config: 这是一个总的配置文件。
        """
        # 导入和分配设置。
        self.config = config
        self.dataloader_config = self.config['torch_dataloaders']
        self.model_config = self.config['torch_models']
        self.trainer_config = self.config['trainer']

        # trainer设置的设置。
        self.callbacks_config = self.trainer_config['callbacks']
        self.loggers_config = self.trainer_config['loggers']
        self.device_config = self.trainer_config['device']

        # 这部分或许在使用LightningDataModule之后可以从构建trainer中分离。
        # 生成train_dataloader和val_dataloader。
        self.train_dataloader, self.val_dataloader = self.build_dataloader()

        # 设置模型。
        self.model = self.build_model()
        # 设置trainer。
        self.callbacks = self.build_callbacks()
        self.loggers = self.build_logger()
        self.record_config()

        self.trainer = self.build_trainer()

    def build_dataloader(self):
        """
        构建dataloader。
        由于这里我使用了工厂模式，这段代码高度复用不需要修改。
        lightning的LightningDataModule也有类似的设计。
        """
        dataloader_factory = DataLoaderFactory(self.dataloader_config)
        train_dataloader = dataloader_factory.get_train_dataloader()
        val_dataloader = dataloader_factory.get_val_dataloader()
        return train_dataloader, val_dataloader

    def build_model(self):
        """构建模型。"""
        model = LightningModel(self.config)
        # 断点续训判断
        return model

    def build_trainer(self, mode: str = 'train') -> pl.Trainer:
        trainer_base_config = {
            'default_root_dir': self.trainer_config['default_root_dir'],
            'max_epochs': self.trainer_config['max_epochs'],
            'callbacks': self.callbacks,
            'logger': self.loggers,
            'accelerator': self.device_config['accelerator'],
            'devices': self.device_config['devices'],
        }
        if mode == 'train':
            # 正常训练。
            pass
        elif mode == 'fast_dev_run':
            # 快速调试。
            trainer_base_config['fast_dev_run'] = True  # 这里也可以传入int，代表快速测试的batch。
        elif mode == 'limit_batches':
            # 限制batch大小调试。
            trainer_base_config['limit_train_batches'] = 0.1
            trainer_base_config['limit_val_batches'] = 0.1
        elif mode == 'num_sanity_val_steps':
            # 提前运行验证步。
            trainer_base_config['num_sanity_val_steps'] = 2
        elif mode == 'bottlenecks':
            # 测试模型各步骤的运行时间。
            trainer_base_config['profiler'] = 'simple'  # or advanced
        trainer = pl.Trainer(**trainer_base_config)
        return trainer

    def build_callbacks(self):
        callback_factory = CallbackFactory(self.callbacks_config)
        callbacks = callback_factory.get_callbacks()
        return callbacks

    def build_logger(self):
        logger_factory = LoggerFactory(self.loggers_config)
        loggers = logger_factory.get_loggers()
        return loggers

    def record_config(self):
        # 将配置文件解析并转换，给所有的logger记录全部的配置。
        config_dict = OmegaConf.to_container(self.config, resolve=True)
        for logger in self.loggers:
            logger.log_hyperparams(config_dict)

    def train(self, mode: str = 'train'):
        """进行训练。"""
        train_base_config = {
            'torch_models': self.model,
            'train_dataloaders': self.train_dataloader,
            'val_dataloaders': self.val_dataloader,
        }
        if mode == 'train':
            self.trainer.fit(**train_base_config)
        elif mode == 'resume':
            train_base_config['ckpt_path'] = ''
            self.trainer.fit(**train_base_config)

