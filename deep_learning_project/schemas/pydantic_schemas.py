"""
基于pydantic构建的schema定义。

注意:
    - 如果使用ExperimentConfig的对象方法model_dump，需要指定`by_alias=True`。
        (model_config为pydantic的预留字段，为了避免崇明，命名为`model_config_`)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class ExperimentConfig(BaseModel):
    """
    总配置文件。
    """
    data_module_config: dict
    model_config_: dict
    trainer_config: dict


class DataModuleConfig(BaseModel):
    ...


class ModelConfig(BaseModel):
    ...


class TrainerConfig(BaseModel):
    ...

