"""
为进行各种操作提供的各种cli的实现。

主要有：
    - argparse
    - hydra
    - LightningCLI
"""

from __future__ import annotations

from argparse import ArgumentParser

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


def cli():
    pass


def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', help='config file', required=True)
    args = parser.parse_args()
    # 然后使用传入的参数


if __name__ == '__main__':
    main()
