# Flash Boilerplate

## 概述
在很多的深度学习任务中，很多操作和流程是重复的。
因此，我基于lightning进行二次开发，构建了一个通用模板。lightning的思想是，约定一些规范的做法，从而简化大量繁琐的工程开发和控制；保留大量的hook，以保证灵活性。
我在此基础上进一步进行了约定，从而再次降低对于模型训练的工程实现。

这个仓库可以作为其他任务的基础模板。复制整个仓库，然后在这个基础上修改和添加。


## 约定
通过各种任务的经验，我总结出一些最佳实践。
在遵循我默认的一些约定的条件下，这个模板的大部分内容并不需要修改。
你可以完全专注于:
- 数据的处理和构建。
- 模型的设计和调优。
- 训练方法的改进。

当然，如果对于具体任务需要，这个模板依然留有进行配置和修改的方法。

## 项目结构
基础的，作为我的通用项目，第一层为 `assets`、`configs`、`deep_learning_project`、`scripts`、`tests`。该仓库特有的为`deep_learning_project`包。关于我的通用项目结构的具体说明可查看()。
### data_processing
对于数据处理，更好的实践是

### deep_learning_project
关于deep-learning的主体部分，包括数据加载、模型构建、模型训练。
相关名称加`torch_`为前缀的原因是`huggingface`的`transformers`相关系列的工具已经使用了例如`datasets`这样的名称，这里为了与我具体工程中的名称做区别。
#### torch_dataloaders
torch中关于数据集加载定义的方法，分为标准的2步分离的dataset和dataloader。关键组成部分为:
- torch_datasets:
- collate_fns:
- torch_dataloaders:

#### torch_models
基于torch定义的模型。只要是torch定义的模型以及相关的扩展库导入的模型，

#### trainer


### configs
在此之前我对于配置的管理方法。这个仓库中的示例是个复杂的用法，实际使用可以简化并以统一配置文件进行管理。

可以参考的做法有:
- 文件格式:
- 管理工具:


## 使用实例
#### 老项目

#### 新项目

