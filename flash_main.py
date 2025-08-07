"""
总入口。在这里启动模型的训练。
"""

from __future__ import annotations

from deep_learning_project.experiment_runner import ExperimentRunner

from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


# 或许每次需要在这里修改配置文件的路径。
@hydra.main(config_path=str(Path(__file__).parent / "configs"), config_name="config")
def main(experiment_config):
    ExperimentRunner.run(
        mode='train',
        experiment_config=experiment_config,
    )


if __name__ == '__main__':
    main()

