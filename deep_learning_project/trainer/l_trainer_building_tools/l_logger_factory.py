"""
构建pl.Trainer可使用的logger。

这个文件几乎不需要修改。

仅功能添加修改该文件:
    - 增加log方法。
    - 使用社区log工具。

预设置各种获取logger的方法，仅需要:
    - 在配置文件中设置。(实际上，仅设置wandb。)
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

    # ====必要方法。====
    @staticmethod
    def create_wandb_logger(
        logger_config: dict,
    ) -> Logger:
        wandb_logger = WandbLogger(
            **logger_config,
        )
        return wandb_logger

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

    # ====预留方法。====
    @staticmethod
    def create_mlflow_logger(
        logger_config: dict,
    ) -> Logger:
        mlflow_logger = MLFlowLogger(
            **logger_config,
        )
        return mlflow_logger

