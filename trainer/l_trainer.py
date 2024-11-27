# from dataloader import
from trainer.l_model import LightningModel

import torch
import lightning as pl

from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra


class LightningTrainer:
    @hydra.main(config_path=str(Path(__file__).parent / "configs"), config_name="config")
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.trainer = pl.Trainer()

        self.model = LightningModel(config)

    def train(self):
        self.trainer.fit(model=self.model)


if __name__ == '__main__':
    pass
