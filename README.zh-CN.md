# Flash Boilerplate
一个基于 PyTorch Lightning 的深度学习项目通用模板。

## 核心理念
在深度学习项目开发中，很多工程任务都是重复且繁琐的。为了解决这个问题，我基于 **PyTorch Lightning** 的核心思想(**约定优于配置**)构建了这个通用模板。

PyTorch Lightning 已经简化了大量的工程开发和训练控制，但为了进一步提升效率，我在其基础上又做了一些额外的约定和封装。

这个模板的核心价值在于:
- **专注于核心任务:** 你可以完全专注于数据处理、模型设计和训练方法的改进，而不用浪费时间从零开始构建项目框架。
- **提高开发效率:** 借助它，可以省去深度学习项目初始阶段的重复性构建，让你直接进入模型开发的环节。
- **可扩展性:** 尽管有很多约定，但模板依然保留了大量的 **hooks**，以满足未来定制化和扩展的需求。

这个仓库精简纯代码约一千多行，是一个轻量且强大的起点，克隆后即可作为新项目的基础。

## 快速入门
### 安装
```bash
git clone https://github.com/yuliu625/Yu-Flash-Boilerplate.git
cd Flash-Boilerplate
# 安装所有依赖
# 参考链接中的 dl_env 配置
```

### 项目结构
```bash
.
├── assets/                  # 通用资产，如文档
├── configs/                 # 配置文件
├── deep_learning_project/   # 深度学习项目主体
│   ├── torch_dataloaders/   # 数据加载器和数据集
│   ├── torch_models/        # 模型定义
│   ├── trainer/             # 训练器相关工具
│   ├── utils/               # 通用工具函数
│   └── schemas/             # Pydantic 数据模式
├── scripts/                 # 运行脚本
├── tests/                   # 测试
└── README.md
```

### 运行
使用 configs 目录下的示例配置文件，通过以下脚本直接运行一个训练任务:
```bash
bash scripts/run.sh
```


## 设计哲学与约定
通过我个人在多个深度学习任务中的经验，我总结出了一些最佳实践，并将它们固化在了这个模板中。如果你遵循这些约定，就可以最大化地利用这个模板的便利性。

### 1. 注册机制: 可配置、可插拔的组件
项目中的所有核心组件都通过**工厂模式**(**Factory Pattern**)和**策略模式**(**Strategy Pattern**)进行封装，并采用注册机制来管理。这意味着可以通过修改配置文件，而**无需修改代码**来切换不同的组件，实现完全的训练配置反序列化。

可注册的组件包括:
- `torch_datasets`: 数据加载方法。
- `torch_models`: 模型定义。
- `loss_fn`: 引导训练的损失函数。
- `torchmetrics_metrics`: 验证模型性能的评价指标。
- `callback`: 训练控制的回调工具。
- `logger`: 日志记录器。

这些组件的具体使用和构建方法，请查看对应的每个包中的 `__init__.py` 的docstring，里面非常详尽地说明实现方法的规范。

### 2. 数据处理约定
为了确保数据流动的统一性，约定所有数据在传递时都使用 `dict` 格式。
- **Dataset 输出**: `dataset` 返回的 `dict` 必须包含 `'target'` 和 `'data'` 两个键。
- **Dataloader 输出**: `dataloader` 返回的 `dict` 也必须包含 `'targets'` 和 `'datas'` 两个键(`collate_fn` 需要进行相应的处理)。

### 3. 配置管理
项目中的所有配置都以 `YAML` 文件形式存储在 `configs` 目录下。
- **单一配置:** 一份配置文件包含一个任务的全部配置，易于版本控制。
- **批量生成:** 使用 `jinja2` 模板引擎，批量生成配置文件，适用于超参数搜索等大规模实验场景。


## 适合场景
### 这个模板非常适合你，如果你的需求是:
- **深度学习网络设计:** 你需要直接设计和修改模型架构。
- **多模态任务:** 你需要自定义精细的跨模态计算。
- **前沿领域探索:** 相关领域缺少现成的工具库，需要从零开始构建。

### 替代方案
在某些情况下，可能存在更适合的工具:
- **Keras:** 如果是简单的入门尝试或概念验证，Keras 是最容易上手的深度学习工具。(但我不喜欢tensorflow)
- **Hugging Face Transformers:** 如果专注于 NLP 任务，Transformers 库提供了与 PyTorch Lightning 类似的设计理念和低耦合的组件，是一个更好的选择。(我很喜欢)

## 链接与参考
你可以通过以下链接，更深入地了解本项目所依赖或相关的工具和理念:
- **通用项目结构:** [Path-Toolkit](https://github.com/yuliu625/Yu-Path-Toolkit)
- **Python 环境配置:** [Python-Environment-Configs](https://github.com/yuliu625/Yu-Python-Environment-Configs)
- **深度学习工具集:** [Deep-Learning-Toolkit](https://github.com/yuliu625/Yu-Deep-Learning-Toolkit)

### 实际应用案例
以下是基于这个模板（或其前身）构建的实际项目，你可以作为参考:
#### 新项目(追求函数式编程与设计模式):
- [Firm-Network-Predictor](https://github.com/yuliu625/Firm-Network-Predictor)
#### 老项目(基于早期面向对象设计):
- [text-classification-trainer](https://github.com/yul1024/text-classification-trainer)
- [image-classification-trainer](https://github.com/yul1024/image-classification-trainer)
- [audio-classification-trainer](https://github.com/yul1024/audio-classification-trainer)

## 未来计划
我计划在未来进一步完善这个模板，包括:
- **集成自动化超参数搜索:** 引入 `optuna` 或 `ray[tune]` 等工具，实现更高级的自动化实验。
- **分布式与并行训练支持:** 进一步利用 `ray` 的能力，简化大规模数据处理和分布式训练。



[//]: # (### data related)

[//]: # (约定:)

[//]: # (- data_processing辅助: 重复以及过于繁重的任务，由data_processing包中的方法实现，)

[//]: # (- data_processor工具类: 数据处理方法定义为静态函数，由额外工具类实现。)

[//]: # (- 例外: 快速验证场景下，以上方法全部绑定在dataset中，后续以该约定进行解耦。)

[//]: # ()
[//]: # ()
[//]: # (dataset和dataset-factory的2种实现方法:)

[//]: # (- 依赖注入:)

[//]: # (    - dataset定义更专注数据处理和加载方法，而不关注数据获取方法。&#40;对于ControlDataset任需要关注。&#41;)

[//]: # (    - dataset-factory或额外构建loading-methods工具类实现数据获取。)

[//]: # (- 数据集类型:)

[//]: # (    - dataset处理全部的工作。&#40;配置文件需要注意签名变量一致性。&#41;)

[//]: # (    - dataset-factory仅传递可序列化参数，无需大规模修改。)

[//]: # ()
[//]: # ()
[//]: # (### data_processing)

[//]: # (对于数据处理，更好的实践是尽可能缓存重复的数据操作。这样的好处是:)

[//]: # (- 速度: 用空间换时间。对于科研任务这很划得来。)

[//]: # (- 版本控制: 可以完全记录各种情况下的数据情况。)

[//]: # ()
[//]: # (当然，这样的挑战是:)

[//]: # (- 需要更好的文件管理策略。&#40;因此我构建了`Path-Toolkit`仓库&#41;)

[//]: # ()
[//]: # (### deep_learning_project)

[//]: # (关于deep-learning的主体部分，包括数据加载、模型构建、模型训练。)

[//]: # (相关名称加`torch_`为前缀的原因是`huggingface`的`transformers`相关系列的工具已经使用了例如`datasets`这样的名称，这里为了与我具体工程中的名称做区别。)

[//]: # ()
[//]: # (#### torch_dataloaders)

[//]: # (torch中关于数据集加载定义的方法，分为标准的2步分离的dataset和dataloader。关键组成部分为:)

[//]: # (- torch_datasets: 基于原生torch的各种场景的dataset的定义。)

[//]: # (- collate_fns: dataloader的重要参数collate_fn。)

[//]: # (- torch_dataloaders: 基于原生torch的dataloader的定义。)

[//]: # ()
[//]: # (这部分每个项目都会因为具体情况不同，而需要独立定义。)

[//]: # ()
[//]: # (#### torch_models)

[//]: # (基于torch定义的模型。只要是torch定义的模型以及相关的扩展库导入的模型，都以factory-pattern定义在这里。并统一以单一的配置项实现具体模型产品的超参数设置。)

[//]: # (我额外构建了`torch_modules`包，这代表模型中有复用或特殊模块的情况。)

[//]: # ()
[//]: # (#### trainer)

[//]: # (为所有任务可复用的trainer构建。大部分方法都已经构造并封装好了，具体任务留有扩展空间。关键组成部分包括:)

[//]: # (- l_model_building_tools: 训练相关的配置。起这个名字是因为对于`lightning`，这些相关的构建是和`LightningModule`绑定的。这里需要注册:)

[//]: # (  - loss_fn_factory: 基于torch.nn.Module的loss_fn。)

[//]: # (  - optimizer_class_factory: 基于torch.optim.Optimizer的optimizer。)

[//]: # (  - metric_factory: 基于torchmetrics.Metric的metrics。)

[//]: # (- l_trainer_building_tools: 训练过程的控制。内容为`lightning`及其社区提供的callback和logger。在这里，我以及预构建好常用的配置。)

[//]: # (- l_data_module: `lightning`中控制data loading的方法。如果符合该项目中`torch_dataloader_factory`实现，这个文件不需要修改。)

[//]: # (- l_model: `lightning`中对模型的定义。如果符合该项目中`torch_model_factory`和`l_model_building_tools`的实现，这个文件不需要修改。)

[//]: # (- l_model_factory: 结合各个工厂实现可序列化配置参数实例化LightningModule对象。不需要修改。)

[//]: # (- l_trainer_builder: 构建训练器的方法。除非增加了callback和logger，否则不需要修改。)

[//]: # ()
[//]: # ()
[//]: # (#### schemas)

[//]: # (基于pydantic定义的数据类。主要对配置文件的合法性进行检验，同时也可用于数据处理过程和模型输出的输出合法性检验。)

[//]: # ()
[//]: # (约定的实践是:)

[//]: # (- 配置定义: )

[//]: # (  - 复杂或多变的配置项不进行检验，仅预留位置。)

[//]: # (  - 固定的配置项我已经构建好，仅进行简单的扩展。)

[//]: # (  - 对于大规模实验，需要完善定义所有的配置。)

[//]: # (- 使用:)

[//]: # (  - 调试中进行schema检验。)

[//]: # (  - 频繁修改架构的情况，为了效率，仅检验部分必要配置。)

[//]: # ()
[//]: # (### configs)

[//]: # (在此之前我对于配置的管理方法。这个仓库中的示例是个复杂的用法，实际使用可以简化并以统一配置文件进行管理。)

[//]: # ()
[//]: # (可以参考的工具有:)

[//]: # (- 文件格式: `yaml`、`json`、`py`。)

[//]: # (- 管理工具: `omegaconf`、`hydra`。)

[//]: # ()
[//]: # (我的实践是:)

[//]: # (- 配置文件以`yaml`构建。)

[//]: # (  - 唯一配置: 一份配置文件包括全部的配置。当然可以使用`hydra`等工具实现配置文件的组合，但是会复杂化配置文件的版本控制。)

[//]: # (  - 关系: 使用占位符语法，重复利用实际一致的配置项。)

[//]: # (  - 批量生成: 对于多配置情况，使用`jinja2`语法定义需要动态改变的配置项，再进行批量生成。非常适合超参数搜索。)

[//]: # (- 日志以`jsonl`实现。)

[//]: # (  - jsonl文件有很多好处，同时也是我主要使用的数据存储的文件格式。)

[//]: # (- 对象以`py`定义。)

[//]: # (  - 有些情况无法避免使用python直接定义的对象会更加方便。)

[//]: # ()
[//]: # (### scripts)

[//]: # (各种运行脚本，包括:)

[//]: # (- 数据处理pipeline。)

[//]: # (- 训练控制。)

[//]: # (- 实验结果分析。)

[//]: # ()
[//]: # (关于脚本的设计，我的规范是不遵循开发和优化原则，完全对修改关闭，以保证这些脚本随时都可以运行以复现某些操作。)

[//]: # ()
[//]: # (### tests)

[//]: # (基于`pytest`的测试工程。已预定义了大量测试代码，可快速检验:)

[//]: # (- 当前服务器可用状态。)

[//]: # (- 当前配置正确性。)


