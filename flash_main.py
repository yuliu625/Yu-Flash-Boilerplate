from trainer import LightningTrainer

from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra


@hydra.main(config_path=str(Path(__file__).parent / "configs"), config_name="config")
def main(config):
    l_trainer = LightningTrainer(config)
    l_trainer.train()


if __name__ == '__main__':
    main()
