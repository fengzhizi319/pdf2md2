"""项目运行配置与默认目录。

这个模块负责两件事：
- 计算仓库根目录下的默认输出路径
- 定义 `PipelineConfig`，把一次 ingest 任务需要的主要参数集中起来

CLI、示例脚本和纯代码调用都会依赖这里的默认值。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# 统一把默认目录锚定到仓库根目录，避免从 IDE、终端或脚本的不同 cwd 启动时
# 把输出写到意料之外的位置。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKDOWN_DIR = PROJECT_ROOT / "pdf"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data/chroma"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "data/manifests"


@dataclass(slots=True)
class PipelineConfig:
    """描述一次 ingest 任务的主要参数。

    CLI、示例脚本和纯代码调用都会先组装这个对象，再交给 `ingest_pdf` 执行，
    这样主流程只有一套实现，入口可以有很多种。
    """

    markdown_dir: Path = DEFAULT_MARKDOWN_DIR
    chroma_dir: Path = DEFAULT_CHROMA_DIR
    manifest_dir: Path = DEFAULT_MANIFEST_DIR
    collection_name: str = "pdf-knowledge-base"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    embedder_type: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    hash_embedding_dimensions: int = 384
