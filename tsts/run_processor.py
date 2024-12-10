"""
实例化一个processor，查看其对于数据的转换。
"""

from dataloader.dataset.processor import Processor

from pathlib import Path


def check_processor(config, data_path_str: str):
    processor = Processor(config)
    data_path = Path(data_path_str)
    result = processor(data_path)
    return result


if __name__ == '__main__':
    """这里需要指定processor的设置，并指定测试数据的路径。"""
    result = check_processor({}, r'')
    print(result)
