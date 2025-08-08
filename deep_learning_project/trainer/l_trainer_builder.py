"""
lightning提供的Trainer。

lightning中Trainer的特点:
    - 独立: 比其他训练框架更加彻底，完全和数据、模型、训练方法无关。这使得这个模块复用性极高。
"""

from __future__ import annotations

from .l_trainer_building_tools import (
    LCallbackFactory,
    LLoggerFactory,
)

import lightning as pl
# from omegaconf import OmegaConf

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.loggers import Logger


class LTrainerBuilder:
    """
    构建pl.Trainer的方法。

    该工具类采用构建者模式实现，所有方法接受可序列化数据构造对象。
    """
    # ====暴露方法.====
    @staticmethod
    def build_lightning_trainer(
        mode: Literal['train', 'fast_dev_run'],
        trainer_config: dict,
    ) -> pl.Trainer:
        # trainer中本身可序列化的参数。
        trainer_base_config = trainer_config['trainer_base_config']
        # 回调相关功能。
        callbacks = LTrainerBuilder.build_callbacks(
            callback_configs=trainer_config['callback_configs'],
        )
        # 日志功能。
        loggers = LTrainerBuilder.build_loggers(
            logger_configs=trainer_config['logger_configs'],
        )
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
            raise NotImplementedError
        elif mode == 'num_sanity_val_steps':
            # 提前运行验证步。
            trainer_base_config['num_sanity_val_steps'] = 2
            raise NotImplementedError
        elif mode == 'bottlenecks':
            # 测试模型各步骤的运行时间。
            trainer_base_config['profiler'] = 'simple'  # or advanced
            raise NotImplementedError
        # 构建trainer对象。
        trainer = pl.Trainer(
            **trainer_base_config,  # 本身可序列化的参数。
            callbacks=callbacks,  # 构建的callback对象。
            logger=loggers,  # 构建的logger对象。
        )
        return trainer

    # ====工具方法。对接方法。====
    @staticmethod
    def build_callbacks(
        callback_configs: list[dict],
    ) -> list[Callback]:
        callbacks = [
            LCallbackFactory.create_callback(
                callback_name=callback_config['callback_name'],
                callback_config=callback_config['callback_config'],
            )
            for callback_config in callback_configs
        ]
        return callbacks

    # ====工具方法。对接方法。====
    @staticmethod
    def build_loggers(
        logger_configs: list[dict],
    ) -> list[Logger]:
        loggers = [
            LLoggerFactory.create_logger(
                logger_name=logger_config['logger_name'],
                logger_config=logger_config['logger_config'],
            )
            for logger_config in logger_configs
        ]
        return loggers

    # def record_config(self):
    #     # 将配置文件解析并转换，给所有的logger记录全部的配置。
    #     config_dict = OmegaConf.to_container(self.config, resolve=True)
    #     for logger in self.loggers:
    #         logger.log_hyperparams(config_dict)

