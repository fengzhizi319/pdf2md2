"""examples 目录共享的调试辅助函数。

这些工具专门服务于 debug 脚本：
- 统一样例 PDF 路径
- 统一打印格式
- 提供小型 demo chunk 数据

这样每个脚本都能专注演示某一个阶段，而不必重复样板代码。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PDF_PATH = PROJECT_ROOT / "pdf/Understanding Lasso – A Novel Lookup Argument Protocol.pdf"


def bootstrap_debug_environment() -> None:
    """让 examples 脚本在 IDE 切换解释器后仍能直接运行。

    - `src` 布局项目在未执行 editable install 时，也能从仓库根目录解析 `pdf2md_rag`
    - macOS + conda 下预先放开 OpenMP 重复加载检查，避免 Marker 首次运行时直接 abort
    """
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    if sys.platform == "darwin" and os.environ.get("CONDA_PREFIX"):
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


bootstrap_debug_environment()

if TYPE_CHECKING:
    from pdf2md_rag.models import Chunk


def print_title(title: str) -> None:
    """打印统一的章节标题，方便在终端里快速分辨脚本输出。"""
    print(f"\n=== {title} ===")


def print_kv(key: str, value) -> None:
    """把调试信息格式化成 key-value 形式，便于肉眼对齐查看。"""
    print(f"{key:<14}: {value}")


def preview_text(text: str, limit: int = 220) -> str:
    """把多行文本压成单行预览，适合在 debug 输出里快速扫一眼。"""
    single_line = " ".join(text.split())
    return single_line[:limit] + ("..." if len(single_line) > limit else "")


def build_demo_chunks() -> list[Chunk]:
    """构造一小组稳定的示例 chunk。

    这些数据不依赖真实 PDF 提取，适合：
    - embedding 调试
    - 向量库调试
    - search / QA 离线调试
    """
    from pdf2md_rag.models import Chunk

    return [
        Chunk(
            chunk_id="demo-1",
            text="Lasso is a lookup argument protocol for efficient verification in proof systems.",
            metadata={
                "source_path": str(PDF_PATH),
                "source_name": PDF_PATH.stem,
                "heading": "Overview",
                "page": 1,
                "chunk_index": 0,
                "text_length": 79,
            },
        ),
        Chunk(
            chunk_id="demo-2",
            text="RAG pipelines split markdown into chunks, embed them, and store them in Chroma.",
            metadata={
                "source_path": str(PDF_PATH),
                "source_name": PDF_PATH.stem,
                "heading": "Implementation",
                "page": 2,
                "chunk_index": 1,
                "text_length": 83,
            },
        ),
    ]
