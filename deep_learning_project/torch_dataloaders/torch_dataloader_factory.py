"""
基于原生torch的dataloader的定义。

注意:
    - 对于dataloader的测试，需要结合collate_fn一同进行。
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    import torch
    from torch.utils.data import Dataset


class DataLoaderFactory:
    """
    生成dataloader的工厂。

    提供的基础方法:
        - create_dataloader: 构建dataloader的方法，规定了需要的参数。

    需要实现的方法:
        - create_train_dataloader
        - create_val_dataloader
    """
    # ====暴露方法。需要实现的方法。====
    @staticmethod
    def create_train_dataloader(
        train_dataset: Dataset,
        collate_fn: Callable[..., torch.Tensor],
        batch_size: int,
        is_shuffle: bool,
        num_workers: int,
        dataloader_kwargs: dict,
    ) -> DataLoader:
        # 设置dataset。
        # dataset的可以以依赖注入的方式进行，也可以在这里构建
        # 设置dataloader。
        train_dataloader = DataLoaderFactory.create_dataloader(
            dataset=train_dataset,
            collate_fn=collate_fn,
            is_shuffle=is_shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            **dataloader_kwargs
        )
        return train_dataloader

    # ====暴露方法。需要实现的方法。====
    @staticmethod
    def create_val_dataloader(
        val_dataset: Dataset,
        collate_fn: Callable[..., torch.Tensor],
        batch_size: int,
        is_shuffle: bool,
        num_workers: int,
        dataloader_kwargs: dict,
    ) -> DataLoader:
        # 设置dataset。
        # dataset的可以以依赖注入的方式进行，也可以在这里构建
        # 设置dataloader。
        val_dataloader = DataLoaderFactory.create_dataloader(
            dataset=val_dataset,
            collate_fn=collate_fn,
            is_shuffle=is_shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            **dataloader_kwargs
        )
        return val_dataloader

    # ====基础方法。====
    @staticmethod
    def create_dataloader(
        dataset: Dataset,
        collate_fn: Callable[..., torch.Tensor],
        batch_size: int,
        is_shuffle: bool,
        num_workers: int,
        dataloader_kwargs: dict,
    ) -> DataLoader:
        # 设置dataloader。
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=is_shuffle,
            num_workers=num_workers,
            **dataloader_kwargs,
        )
        return dataloader

