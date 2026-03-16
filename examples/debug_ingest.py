"""调试 `ingest_pdf` 的完整示例脚本。

适合学习：
- `PipelineConfig` 是如何组装的
- 主流程最终会产出哪些文件
- `IngestResult` 里有哪些关键信息

这是最适合在 IDE 里打断点、沿着主流程往下跳转的入口。
"""

from pathlib import Path

from _debug_common import PDF_PATH

from pdf2md_rag.config import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_MARKDOWN_DIR,
    PipelineConfig,
)
from pdf2md_rag.pipeline import ingest_pdf


def main() -> None:
    # 用 hash embedder 作为默认调试模式，避免为了学习代码先下载 embedding 模型。
    config = PipelineConfig(
        markdown_dir=DEFAULT_MARKDOWN_DIR,
        chroma_dir=DEFAULT_CHROMA_DIR,
        manifest_dir=DEFAULT_MANIFEST_DIR,
        collection_name="understanding-lasso-hash-debug",
        embedder_type="hash",
        embedding_model="unused",
        hash_embedding_dimensions=384,
        chunk_size=1200,
        chunk_overlap=200,
    )

    result = ingest_pdf(PDF_PATH, config)

    markdown_path = Path(result.markdown_path).resolve()
    manifest_path = Path(result.manifest_path).resolve()

    print("=== DEBUG INGEST RESULT ===")
    print(f"PDF       : {result.pdf_path}")
    print(f"Markdown  : {markdown_path} | exists={markdown_path.exists()}")
    print(f"Manifest  : {manifest_path} | exists={manifest_path.exists()}")
    print(f"Collection: {result.collection_name}")
    print(f"Pages     : {result.page_count}")
    print(f"Chunks    : {result.chunk_count}")
    print(f"Vectors   : {result.vector_count}")

    if not markdown_path.exists() or not manifest_path.exists():
        raise RuntimeError("Expected output files were not created.")


if __name__ == "__main__":
    main()

