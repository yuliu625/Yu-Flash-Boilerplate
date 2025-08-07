"""
基于dataclass构建的schema定义。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lightning import LightningDataModule, LightningModule
    from pathlib import Path


"""<!--data-module--start-->"""


@dataclass
class LightningDataModuleConfig:
    train_dataset_config: TorchDatasetConfig
    train_dataloader_config: TorchDataloaderConfig
    val_dataset_config: TorchDatasetConfig
    val_dataloader_config: TorchDataloaderConfig


@dataclass
class TorchDatasetConfig:
    ...


@dataclass
class TorchDataloaderConfig:
    ...


"""<!--data-module--end-->"""
"""<!--model--start>"""


@dataclass
class LightningModelConfig:
    ...


@dataclass
class TorchLossFn:
    loss_fn_name: str
    loss_fn_kwargs: dict


@dataclass
class TorchMetric:
    metric_name: str
    metric_kwargs: dict


"""<!--model-end-->"""
"""<!--trainer--start-->"""


@dataclass
class LightningTrainerConfig:
    default_root_dir: str | Path


@dataclass
class LightningLoggerConfig:
    ...


@dataclass
class LightningCallbackConfig:
    ...


"""<!--trainer--end-->"""

