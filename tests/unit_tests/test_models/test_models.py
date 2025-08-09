"""
测试整体模型的输入输出形状是否复合预期。

注意:
    - 对于module的测试不需要使用dataloader，仅需输入预期形状的tensor。
"""

from __future__ import annotations
import pytest

from deep_learning_project.torch_models import TorchModelFactory

import torch

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class TestTorchModels:
    def test_torch_models(self, torch_model_name, torch_model_config):
        torch_model = TorchModelFactory.create_torch_model(
            torch_model_name=torch_model_name,
            torch_model_config=torch_model_config,
        )
        # torch.Size([1, 19, 3, 224, 224])
        example_input_array = torch.rand(1, 5, 3, 224, 224)
        outputs = torch_model(example_input_array)
        print(outputs)
        print(outputs.shape)

