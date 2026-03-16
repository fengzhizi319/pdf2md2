"""项目共享数据模型。

这个文件只定义跨模块传递的核心对象，不承载业务逻辑：
- `MarkdownDocument`：PDF 提取阶段的输出
- `Chunk`：切块阶段的输出

读源码时可以把它当成整条数据流的“数据契约”起点。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MarkdownDocument:
    """PDF 提取阶段的标准输出。

    之所以单独建一个数据类，而不是只返回字符串，是为了让后续切块、落盘、
    manifest 生成都能稳定拿到来源路径和页数等上下文信息。
    """

    source_path: Path
    text: str
    page_count: int


@dataclass(slots=True)
class Chunk:
    """切块阶段的标准输出。

    `text` 是真正参与 embedding 的内容；`metadata` 则保留来源文件、章节、页码、
    chunk 序号等检索时很有用的上下文信息。
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any]
