"""
lightning提供的Trainer。

这个boilerplate最重要的部分。
封装大量的重复配置项。几乎不用重写。
"""

from __future__ import annotations

from .l_trainer_building_tools.l_callback_factory import LCallbackFactory
from .l_trainer_building_tools.l_logger_factory import LLoggerFactory

import lightning as pl
from omegaconf import OmegaConf

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.loggers import Logger


class LightningTrainerBuilder:
    """
    构建pl.Trainer的方法。
    """
    def __init__(
        self,
        trainer_config: dict,
    ):
        """
        :param config: 这是一个总的配置文件。
        """
        # 导入和分配设置。
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

        self.trainer = self.build_lightning_trainer()

    # ====暴露方法.====
    @staticmethod
    def build_lightning_trainer(
        mode: str = 'train',
    ) -> pl.Trainer:
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

    # ====工具方法。====
    @staticmethod
    def build_callbacks(
        callback_configs: list[dict],
    ) -> list[Callback]:
        callbacks = [
            LCallbackFactory.create_callback(
                callback_name=callback_config['callback_name'],
                callback_kwargs=callback_config['callback_kwargs'],
            )
            for callback_config in callback_configs
        ]
        return callbacks

    # ====工具方法。====
    @staticmethod
    def build_loggers(
        logger_configs: list[dict],
    ) -> list[Logger]:
        loggers = [
            LLoggerFactory.create_logger(
                logger_name=logger_config['logger_name'],
                logger_kwargs=logger_config['logger_kwargs'],
            )
            for logger_config in logger_configs
        ]
        return loggers

    def record_config(self):
        # 将配置文件解析并转换，给所有的logger记录全部的配置。
        config_dict = OmegaConf.to_container(self.config, resolve=True)
        for logger in self.loggers:
            logger.log_hyperparams(config_dict)

    # TODO: 待修改。执行训练的代码不和trainer绑定在一起。
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

