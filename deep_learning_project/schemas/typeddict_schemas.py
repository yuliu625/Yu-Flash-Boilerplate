"""
基于TypedDict构建的schema定义。
"""

from __future__ import annotations

from pathlib import Path

from typing import TYPE_CHECKING, TypedDict
# if TYPE_CHECKING:


class LightningDataModuleConfig(TypedDict):
    train_dataset_config: TorchDatasetConfig
    train_dataloader_config: TorchDataloaderConfig
    val_dataset_config: TorchDatasetConfig
    val_dataloader_config: TorchDataloaderConfig


class TorchDatasetConfig(TypedDict):
    ...


class TorchDataloaderConfig(TypedDict):
    ...


"""<!--data-module--end-->"""
"""<!--model--start>"""


class LightningModelConfig(TypedDict):
    ...


class TorchLossFn(TypedDict):
    loss_fn_name: str
    loss_fn_config: dict


class TorchMetric(TypedDict):
    metric_name: str
    metric_config: dict


"""<!--model-end-->"""
"""<!--trainer--start-->"""


class LightningTrainerConfig(TypedDict):
    default_root_dir: str | Path


class LightningLoggerConfig(TypedDict):
    ...


class LightningCallbackConfig(TypedDict):
    ...


"""<!--trainer--end-->"""
