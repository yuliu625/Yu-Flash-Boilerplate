"""
执行训练过程，使用trainer中构建的对象。
"""

from __future__ import annotations

from .trainer.l_data_module import LDataModule
from .trainer.l_model_factory import LModelFactory
from .trainer.l_trainer_builder import LTrainerBuilder

from omegaconf import OmegaConf

from typing import TYPE_CHECKING, Literal
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
    mode: Literal['train', 'fast_dev_run'],
    trainer_config: dict,
) -> Trainer:
    trainer = LTrainerBuilder.build_lightning_trainer(
        mode=mode,
        trainer_config=trainer_config,
    )
    return trainer


def main(
    mode: Literal['train', 'fast_dev_run', 'resume'],
    experiment_config: dict,
):
    data_module = get_data_module(
        data_module_config=experiment_config['data_module_config'],
    )
    model = get_model(
        model_config=experiment_config['model_config'],
    )
    trainer = get_trainer(
        mode=mode,
        trainer_config=experiment_config['trainer_config'],
    )
    # 进行训练
    trainer.fit(
        model=model,
        datamodule=data_module,
    )


if __name__ == '__main__':
    mode_ = r'train'
    experiment_config_ = {}

    main(
        mode=mode_,
        experiment_config=experiment_config_,
    )

