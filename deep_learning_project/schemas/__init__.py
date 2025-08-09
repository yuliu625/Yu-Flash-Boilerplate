"""
一些数据类的定义，用于规范化输入输出。

特性:
    - 工具: 基于pydantic的schema定义。
    - 兼容性: 为了保证兼容性和迁移，pydantic的BaseModel对象会被转换为dict进行使用。
    - 非强制: 可以不进行schema检验，以适应快速开发和低要求场景。
    - 其他方案: dataclass和TypedDict。
"""

