"""
构建lightning trainer需要的callback。

统一工厂方法构建callback，具体设置由配置文件实现。

重要callback:
    - model_checkpoint:
    - early_stopping:
功能callback:
    - model_summary:
    - device_stats_monitor:
"""

from __future__ import annotations

from lightning.pytorch.callbacks import (
    ModelSummary,
    DeviceStatsMonitor,
    ModelCheckpoint,
    EarlyStopping,
)

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback


class LCallbackFactory:
    # ====暴露方法。====
    @staticmethod
    def create_callback(
        callback_name: Literal[
            'model_checkpoint', 'early_stopping',
            'model_summary', 'device_stats_monitor',
        ],
        callback_config: dict,
    ) -> Callback:
        if callback_name == 'model_checkpoint':
            return LCallbackFactory.create_model_checkpoint_callback(callback_config=callback_config)
        elif callback_name == 'early_stopping':
            return LCallbackFactory.create_early_stopping_callback(callback_config=callback_config)
        elif callback_name == 'model_summary':
            return LCallbackFactory.create_model_summary_callback(callback_config=callback_config)
        elif callback_name == 'device_stats_monitor':
            return LCallbackFactory.create_device_stats_monitor_callback(callback_config=callback_config)

    # ====必要方法。====
    @staticmethod
    def create_model_checkpoint_callback(
        callback_config: dict,
    ) -> Callback:
        """
        checkpoint管理。

        本地持久化checkpoint。lightning构建的checkpoint会自动保存所有所需的内容，并自动处理分布式环境。

        Args:
            callback_config:

        Returns:

        """
        model_checkpoint_callback = ModelCheckpoint(
            **callback_config,
        )
        return model_checkpoint_callback

    # ====必要方法。====
    @staticmethod
    def create_early_stopping_callback(
        callback_config: dict,
    ) -> Callback:
        """
        早停。

        监视训练过程的指定指标，当连续持续恶化，提前终止训练进程。

        Args:
            callback_config:

        Returns:

        """
        early_stopping_callback = EarlyStopping(
            **callback_config,
        )
        return early_stopping_callback

    @staticmethod
    def create_model_summary_callback(
        callback_config: dict,
    ) -> Callback:
        model_summary_callback = ModelSummary(
            **callback_config,
        )
        return model_summary_callback

    @staticmethod
    def create_device_stats_monitor_callback(
        callback_config: dict,
    ) -> Callback:
        device_stats_monitor_callback = DeviceStatsMonitor(
            **callback_config,
        )
        return device_stats_monitor_callback

