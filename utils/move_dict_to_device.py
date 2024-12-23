"""
因为pytorch_lightning.utilities中有move_data_to_device这个方法，后续我再没有使用过我写的这个方法。
后来lightning可以自动处理嵌套数据结构，对分布式环境非常有效，我便再没有使用过相关方法。
"""

import torch


def move_batch_to_device(batch, device):
    """如果数据是dict，然后需要将里面的tensor移动到gpu，用这个方法。"""
    """默认可以发生在：定义好dataset在collate_fn时，在model最开始的forward。"""
    return {key: value.to(device) for key, value in batch.items()}


if __name__ == '__main__':
    pass
