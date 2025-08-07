"""
构建lightning trainer需要的callback。

构造这个的原因是：callback太多了，统一构造会更好。
这个文件还是每次修改一下，通过配置文件似乎没有必要。
"""

from __future__ import annotations

from lightning.pytorch.callbacks import (
    ModelSummary,
    DeviceStatsMonitor,
    ModelCheckpoint,
    EarlyStopping,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback


class CallbackFactory:
    def __init__(self, config: dict):
        self.config = config
        self.choice = self.config['choice']

        # 会不断设置并最终返回的list的callback。
        self.callbacks = []

        self.add_callback(self.choice)

    def add_callback(self, choice: list[str]):
        if 'early_stopping' in choice:
            early_stop_callback = self.get_early_stop_callback()
            self.callbacks.append(early_stop_callback)

    def get_callbacks(self):
        return self.callbacks

    def get_model_summary(self):
        """
        打印模型情况。
        """
        # model_summary = ModelSummary(torch_models, max_depth=-1)
        # return model_summary

    def get_model_checkpoint(self):
        """
        进行checkpoint具体设置.
        """
        model_checkpoint = ModelCheckpoint(
            save_top_k=5,
            monitor='val_f1_score',
            mode='max',
            filename="{epoch:02d}-{val_f1_score:.2f}"
        )
        return model_checkpoint

    def get_early_stop_callback(self):
        """
        设置提前停止。
        监控某个日志指标，根据其变化提前停止训练。
            - 一些实验，可以设置很大的epoch然后提前停止。
            - 另一些实验，使用相同的epoch而不设置这个callback。
        """
        early_stop_callback = EarlyStopping(
            monitor='val_f1_score',
            mode='min',
            patience=5,
        )
        return early_stop_callback


class LCallbackFactory:
    @staticmethod
    def create_callback(
        callback_name: str,
        callback_kwargs: dict,
    ) -> Callback:
        ...

    @staticmethod
    def create_model_checkpoint_callback(
        callback_kwargs: dict,
    ) -> Callback:
        model_checkpoint_callback = ModelCheckpoint(
            **callback_kwargs,
        )
        return model_checkpoint_callback

    @staticmethod
    def create_early_stopping_callback(
        callback_kwargs: dict,
    ) -> Callback:
        early_stopping_callback = EarlyStopping(
            **callback_kwargs,
        )
        return early_stopping_callback

    @staticmethod
    def create_model_summary_callback(
        callback_kwargs: dict,
    ) -> Callback:
        model_summary_callback = ModelSummary(
            **callback_kwargs,
        )
        return model_summary_callback

    @staticmethod
    def create_device_stats_monitor_callback(
        callback_kwargs: dict,
    ) -> Callback:
        device_stats_monitor_callback = DeviceStatsMonitor(
            **callback_kwargs,
        )
        return device_stats_monitor_callback

