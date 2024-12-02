"""
用来测试数据和模型部分的正常运行的工具。
类似单元测试用例，会有notebook形式文件。
数据：
    - 对于数据处理的测试。
    - 对于dataset的测试。
    - 对于dataloader的测试。
模型：
    - 单独对于具体nn.Module的测试，项目中会分为4中形式：
        - lightning自动使用的example_input。
        - nn.Module中写的assert tensor.shape
        - 每个py文件的__main__会快捷执行的测试。
        - 当前这个包。
"""

# 这个包不用对外暴露。仅包内进行测试。
# __all__ = []


# from . import
