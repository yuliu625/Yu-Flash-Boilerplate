"""
测试配置文件的正确。
"""

from __future__ import annotations
import pytest

from deep_learning_project.schemas.pydantic_schemas import (
    ExperimentConfig,
    DataModuleConfig,
    ModelConfig,
    TrainerConfig,
)

import hydra
from omegaconf import OmegaConf
from pprint import pprint
from pathlib import Path

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class TestConfigSchema:
    def test_experiment_config_schema(self, experiment_config_path):
        ...


# @hydra.main(config_path='conf', config_name='test_config')
# def print_config(config):
#     # 打印解析后的配置文件，可以检查配置文件的原本的情况。
#     pprint(OmegaConf.to_yaml(config, resolve=True))
#     # 打印对象格式的配置文件，方便在每一步测试。
#     pprint(OmegaConf.to_object(config))

