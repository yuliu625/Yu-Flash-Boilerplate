"""
基于原生torch的dataset的所相关的实现。

封装了原始各种格式数据的加载，会在torch_dataloaders中使用。

约定:
    - data_processing辅助: 重复以及过于繁重的任务，由data_processing包中的方法实现，
    - data_processor工具类: 数据处理方法定义为静态函数，由额外工具类实现。
    - 例外: 快速验证场景下，以上方法全部绑定在dataset中，后续以该约定进行解耦。

schema和结构约定:
    - dict: 传递数据以dict进行传递。
    - dataset-key: dataset返回的dict的keys=['target','data']。
    - dataloader-key: dataloader返回的dict的keys=['target','data']。(也是collate_fn需要进行的映射处理约定名称。)

dataset和dataset-factory的2种实现方法:
    - 依赖注入:
        - dataset定义更专注数据处理和加载方法，而不关注数据获取方法。(对于ControlDataset任需要关注。)
        - dataset-factory或额外构建loading-methods工具类实现数据获取。
    - 数据集类型:
        - dataset处理全部的工作。(配置文件需要注意签名变量一致性。)
        - dataset-factory仅传递可序列化参数，无需大规模修改。
"""

from .torch_dataset_factory import TorchDatasetFactory
