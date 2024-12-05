"""
总入口。在这里启动模型的训练。
"""

from trainer import LightningTrainer

from pathlib import Path
import hydra


# 或许每次需要在这里修改配置文件的路径。
@hydra.main(config_path=str(Path(__file__).parent / "configs"), config_name="config")
def main(config):
    l_trainer = LightningTrainer(config)
    l_trainer.train()


if __name__ == '__main__':
    main()
