"""
在分类任务下定义的label_map。

可以进行的改进:
    - label_map以配置文件进行指定。
    - 在数据处理阶段就处理得到label_map，而不是在每一遍dataset对象加载阶段进行处理。
以此改进，不需要额外定义这样一个文件。
"""

label_map = dict(

)

