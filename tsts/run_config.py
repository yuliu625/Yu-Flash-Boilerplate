"""
测试配置文件的层级、继承、覆盖关系是否正确。
"""

import hydra
from omegaconf import OmegaConf


@hydra.main(config_path='conf', config_name='test_config')
def test_config(config):
    print(OmegaConf.to_yaml(config, resolve=True))


if __name__ == '__main__':
    test_config()
