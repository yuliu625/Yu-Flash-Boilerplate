"""
实例化一个dataset，查看数据输出的情况。

在具体写dataset时，在对应文件可先查看path的映射情况。
"""

from deep_learning_project.torch_dataloaders.torch_datasets import DFDataset


def get_dataset(config):
    """获得一个实例化的dataset。"""
    return DFDataset(config)


def check_dataset(config):
    """检测dataset的输出情况。"""
    # 根据配置设置，进行实例化。
    common_dataset = get_dataset(config)
    # 打印输出。
    print(len(common_dataset))
    print(common_dataset[0])
    print(common_dataset[1])


if __name__ == '__main__':
    """这里指定config。之后会改进为从命令行或配置文件自动读取。"""
    check_dataset({})
