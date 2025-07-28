"""
数据预处理的部分。

这个工具类构建的目的是:
    - 过于繁重和重复的数据预处理不应该由dataset来进行，而是提前处理好。
    - 以pipeline管理数据处理，保证数据处理的效率和统一。
"""

from __future__ import annotations

from typing import TYPE_CHECKING
# if TYPE_CHECKING:

