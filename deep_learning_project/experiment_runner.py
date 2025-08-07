"""
执行训练过程，使用trainer中构建的对象。
"""

from __future__ import annotations

from .trainer.l_data_module import LDataModule
from .trainer.l_model_factory import LModelFactory
from .trainer.l_trainer_builder import LTrainerBuilder

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from lightning import LightningDataModule, LightningModule, Trainer


class ExperimentRunner:
    @staticmethod
    def get_data_module(
        data_module_config: dict,
    ) -> LightningDataModule:
        data_module = LDataModule(**data_module_config)
        return data_module

    @staticmethod
    def get_model(
        model_config: dict,
    ) -> LightningModule:
        l_model = LModelFactory.create_lightning_model(**model_config)
        return l_model

    @staticmethod
    def get_trainer(
        mode: Literal['train', 'fast_dev_run'],
        trainer_config: dict,
    ) -> Trainer:
        trainer = LTrainerBuilder.build_lightning_trainer(
            mode=mode,
            trainer_config=trainer_config,
        )
        return trainer

    @staticmethod
    def run(
        mode: Literal['train', 'fast_dev_run'],
        experiment_config: dict,
    ):
        data_module = ExperimentRunner.get_data_module(
            data_module_config=experiment_config['data_module_config'],
        )
        model = ExperimentRunner.get_model(
            model_config=experiment_config['model_config'],
        )
        trainer = ExperimentRunner.get_trainer(
            mode=mode,
            trainer_config=experiment_config['trainer_config'],
        )
        # 进行训练
        trainer.fit(
            model=model,
            datamodule=data_module,
        )

