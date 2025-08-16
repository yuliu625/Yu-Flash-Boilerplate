"""
主体关于deeplearning的部分。

定义的具体约定:
    - data:
        - dict: 传递数据以dict进行传递。
        - dataset-key: dataset返回的dict的keys=['target','data']。
        - dataloader-key: dataloader返回的dict的keys=['target','data']。(也是collate_fn需要进行的映射处理约定名称。)
    - model:
        - torch_model_config: 模型的实例化由可序列化的数据完成，统一封装在dict中，不强制进行schema检验。
    - trainer:

需要进行注册的部分:
    - torch_dataloaders:
        - torch_datasets:
            - torch_datasets.py: 定义各种dataset。
                复杂情况，额外构建data_processor工具类辅助数据处理。(快捷开发，可以直接定义为对应dataset的方法。)
            - torch_dataset_factory.py: 注册在torch_datasets.py中定义的dataset类。
                dataset和dataset-factory有2种实现关系，目前的实现方法为依赖注入。
        - collate_fn_factory.py: 定义并注册collate_fn。(函数方法，直接写在同一个文件中。)(复杂情况可额外构建一个文件导入。)
    - torch_models:
        - torch_models.py: 定义各种models。
        - torch_model_factory.py: 注册在torch_models.py种定义模型，并执行必要的操作。(例如参数冻结。)
            - torch_modules: 定义复用性组件。
    - trainer:
        - l_model_building_tools: 定义并注册所有deep-learning-model训练必要的组件。
        - l_model: 复杂训练方法，修改运行步。
"""

from .experiment_runner import ExperimentRunner

