"""
测试配置文件的层级、继承、覆盖关系是否正确。
"""

import hydra
from omegaconf import OmegaConf

from pprint import pprint


@hydra.main(config_path='conf', config_name='test_config')
def print_config(config):
    # 打印解析后的配置文件，可以检查配置文件的原本的情况。
    pprint(OmegaConf.to_yaml(config, resolve=True))
    # 打印对象格式的配置文件，方便在每一步测试。
    pprint(OmegaConf.to_object(config))


if __name__ == '__main__':
    print_config()
