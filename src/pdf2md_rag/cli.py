"""命令行入口层。

这个文件本身不实现核心算法，它的职责是：
- 解析命令行参数
- 组装 `PipelineConfig`
- 调用共享的 ingest / query 逻辑
- 打印结果并做关键输出校验

如果想学主流程实现，优先去看 `pipeline.py`；如果想学项目如何暴露给用户，就看这里。
"""

from __future__ import annotations

from pathlib import Path

import typer

from .config import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_MARKDOWN_DIR,
    PipelineConfig,
)
from .embeddings import build_embedder
from .pipeline import ingest_pdf
from .vectorstore import query_collection

app = typer.Typer(help="Build and query a local PDF RAG knowledge base.")


def _verify_output_file(label: str, file_path: str | Path) -> Path:
    """校验某个关键输出文件是否真的落盘。"""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        typer.echo(f"ERROR: {label} file was not created: {path}")
        raise typer.Exit(code=1)
    typer.echo(f"Verified {label}: {path}")
    return path


@app.command()
def ingest(
    pdf_path: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    collection: str = typer.Option("pdf-knowledge-base", help="Chroma collection name."),
    markdown_dir: Path = typer.Option(DEFAULT_MARKDOWN_DIR, help="Directory to save Markdown output."),
    chroma_dir: Path = typer.Option(DEFAULT_CHROMA_DIR, help="Directory to persist Chroma data."),
    manifest_dir: Path = typer.Option(DEFAULT_MANIFEST_DIR, help="Directory to save manifest JSON."),
    chunk_size: int = typer.Option(1200, min=200, help="Target chunk size in characters."),
    chunk_overlap: int = typer.Option(200, min=0, help="Chunk overlap in characters."),
    embedder: str = typer.Option("sentence-transformers", help="Embedding backend: sentence-transformers or hash."),
    embedding_model: str = typer.Option("BAAI/bge-small-en-v1.5", help="Sentence Transformers model name."),
    batch_size: int = typer.Option(32, min=1, help="Reserved batch size config for future extension."),
) -> None:
    """CLI 入口：构造配置并调用共享 ingest 主流程。"""
    config = PipelineConfig(
        markdown_dir=markdown_dir,
        chroma_dir=chroma_dir,
        manifest_dir=manifest_dir,
        collection_name=collection,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedder_type=embedder,
        embedding_model=embedding_model,
        batch_size=batch_size,
    )
    result = ingest_pdf(pdf_path=pdf_path, config=config)
    markdown_path = _verify_output_file("Markdown", result.markdown_path)
    manifest_path = _verify_output_file("Manifest", result.manifest_path)

    typer.echo(f"Ingested: {result.pdf_path}")
    typer.echo(f"Markdown: {markdown_path}")
    typer.echo(f"Manifest: {manifest_path}")
    typer.echo(f"Collection: {result.collection_name}")
    typer.echo(f"Pages: {result.page_count} | Chunks: {result.chunk_count} | Stored vectors: {result.vector_count}")


@app.command()
def query(
    question: str = typer.Option(..., help="Question to search in the vector store."),
    collection: str = typer.Option("pdf-knowledge-base", help="Chroma collection name."),
    chroma_dir: Path = typer.Option(DEFAULT_CHROMA_DIR, help="Directory where Chroma is persisted."),
    embedder: str = typer.Option("sentence-transformers", help="Embedding backend: sentence-transformers or hash."),
    embedding_model: str = typer.Option("BAAI/bge-small-en-v1.5", help="Sentence Transformers model name."),
    top_k: int = typer.Option(5, min=1, max=20, help="How many matches to return."),
) -> None:
    """CLI 入口：做一次最小向量检索并把结果打印出来。"""
    query_embedder = build_embedder(embedder_type=embedder, model_name=embedding_model)
    rows = query_collection(
        question=question,
        query_embedding=query_embedder.embed_query(question),
        persist_directory=chroma_dir,
        collection_name=collection,
        top_k=top_k,
    )
    if not rows:
        typer.echo("No matches found.")
        raise typer.Exit(code=0)

    for index, row in enumerate(rows, start=1):
        metadata = row["metadata"] or {}
        typer.echo(f"[{index}] distance={row['distance']:.4f} page={metadata.get('page')} heading={metadata.get('heading')}")
        typer.echo(row["document"][:600])
        typer.echo("-" * 80)


if __name__ == "__main__":
    app()
