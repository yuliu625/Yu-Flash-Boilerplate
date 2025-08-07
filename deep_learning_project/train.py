"""
执行训练过程，使用trainer中构建的对象。
"""

from __future__ import annotations

from .trainer.l_data_module import LDataModule

from omegaconf import OmegaConf

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lightning import LightningDataModule, LightningModule, Trainer
    from omegaconf import DictConfig


def get_data_module(
    data_module_config: dict,
) -> LightningDataModule:
    data_module = LDataModule(**data_module_config)
    return data_module


def get_model(
    model_config: dict,
) -> LightningModule:
    ...


def get_trainer(
    trainer_config: dict,
) -> Trainer:
    ...


def main(
    experiment_config: dict,
):
    ...


if __name__ == '__main__':
    pass

