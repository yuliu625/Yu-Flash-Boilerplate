# from dataloader import
from trainer.l_model import LightningModel

import torch
import lightning as pl

from pathlib import Path
from omegaconf import OmegaConf
import hydra


class LightningTrainer:
    def __init__(self, config):
        self.config = config
        self.trainer = pl.Trainer()

        self.model = LightningModel(config)

    def train(self):
        self.trainer.fit(model=self.model)


if __name__ == '__main__':
    pass
