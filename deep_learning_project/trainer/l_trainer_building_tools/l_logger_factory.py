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


class LLoggerFactory:
    # ====暴露方法。====
    @staticmethod
    def create_logger(
        logger_name: Literal[
            'csv', 'tensorboard',
            'wandb', 'mlflow',
        ],
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

    # ====备用方法。====
    @staticmethod
    def create_csv_logger(
        logger_config: dict,
    ) -> Logger:
        csv_logger = CSVLogger(
            **logger_config,
        )
        return csv_logger

    # ====备用方法。====
    @staticmethod
    def create_tensorboard_logger(
        logger_config: dict,
    ) -> Logger:
        tensorboard_logger = TensorBoardLogger(
            **logger_config,
        )
        return tensorboard_logger

    # ====必要方法。====
    @staticmethod
    def create_wandb_logger(
        logger_config: dict,
    ) -> Logger:
        wandb_logger = WandbLogger(
            **logger_config,
        )
        return wandb_logger

    # ====预留方法。====
    @staticmethod
    def create_mlflow_logger(
        logger_config: dict,
    ) -> Logger:
        mlflow_logger = MLFlowLogger(
            **logger_config,
        )
        return mlflow_logger

