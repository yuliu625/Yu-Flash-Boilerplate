# tool chain


## main
基于pytorch进行优化。
### 
- torchmetrics: metrics不应该自己去写，除非很有必要。不再使用sklearn中的方法。
- lightning: 使用更好的trainer，而不用零散的自己去定义train方法。


## 管理
测试工程
- 使用assert。确定每部分输入输出的shape。
- 测试工程。每部分module进行单独测试，有拟定的测试输入。


## 日志
多个日志系统。可视化。


## 配置
- omegaconf
- hydra


