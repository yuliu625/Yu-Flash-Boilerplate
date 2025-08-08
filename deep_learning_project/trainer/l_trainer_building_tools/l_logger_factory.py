"""
构建pl.Trainer可使用的logger。

构造这个的原因是：callback太多了，统一构造会更好。
这个文件还是每次修改一下，通过配置文件似乎没有必要。
"""

from __future__ import annotations

from lightning.pytorch.loggers import (
    CSVLogger,
    TensorBoardLogger,
    WandbLogger,
    MLFlowLogger,
)

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from lightning.pytorch.loggers import Logger


class LoggerFactory:
    def __init__(self, config: dict):
        # 导入配置
        self.config = config
        self.choice = self.config['choice']
        self.wandb_config = self.config['wandb']

        # 会不断设置并最终返回的list的logger。
        self.loggers = []

        # 添加logger
        self.add_logger(self.choice)

    def add_logger(self, choice: list[str]):
        if 'csv' in choice:
            self.loggers.append(self.get_csv_logger())
        if 'tensorboard' in choice:
            self.loggers.append(self.get_tensorboard_logger())
        if 'wandb' in choice:
            self.loggers.append(self.get_wandb_logger())

    def get_loggers(self):
        return self.loggers

    def get_csv_logger(self):
        csv_logger = CSVLogger(save_dir='csv_logs')
        return csv_logger

    def get_tensorboard_logger(self):
        tensorboard_logger = TensorBoardLogger(save_dir='tensorboard_logs')
        return tensorboard_logger

    def get_wandb_logger(self):
        wandb_logger = WandbLogger(
            project=self.wandb_config['project'],
            name=self.wandb_config['name'],
        )
        return wandb_logger


class LLoggerFactory:
    # ====暴露方法。====
    @staticmethod
    def create_logger(
        logger_name: Literal['csv', 'tensorboard', 'wandb', 'mlflow',],
        logger_config: dict,
    ) -> Logger:
        if logger_name == 'csv':
            return LLoggerFactory.create_csv_logger(logger_config=logger_config)
        elif logger_name == 'tensorboard':
            return LLoggerFactory.create_tensorboard_logger(logger_config=logger_config)
        elif logger_name == 'wandb':
            return LLoggerFactory.create_wandb_logger(logger_config=logger_config)
        elif logger_name == 'mlflow':
            return LLoggerFactory.create_mlflow_logger(logger_config=logger_config)

    @staticmethod
    def create_csv_logger(
        logger_config: dict,
    ) -> Logger:
        csv_logger = CSVLogger(
            **logger_config,
        )
        return csv_logger

    @staticmethod
    def create_tensorboard_logger(
        logger_config: dict,
    ) -> Logger:
        tensorboard_logger = TensorBoardLogger(
            **logger_config,
        )
        return tensorboard_logger

    @staticmethod
    def create_wandb_logger(
        logger_config: dict,
    ) -> Logger:
        wandb_logger = WandbLogger(
            **logger_config,
        )
        return wandb_logger

    @staticmethod
    def create_mlflow_logger(
        logger_config: dict,
    ) -> Logger:
        mlflow_logger = MLFlowLogger(
            **logger_config,
        )
        return mlflow_logger

