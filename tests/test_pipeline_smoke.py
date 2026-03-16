from pathlib import Path

from pdf2md_rag.config import PipelineConfig
from pdf2md_rag.models import MarkdownDocument
from pdf2md_rag.pipeline import ingest_pdf


def test_ingest_pdf_end_to_end_with_hash_embedder(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")

    def fake_extract_markdown(path: str | Path) -> MarkdownDocument:
        return MarkdownDocument(
            source_path=Path(path),
            page_count=2,
            text=(
                "# Lasso\n\n"
                "Lasso lookup protocol makes lookup arguments efficient.\n\n"
                "## Details\n\n"
                "This second page mentions RAG chunking and vector storage."
            ),
        )

    monkeypatch.setattr("pdf2md_rag.pipeline.extract_markdown", fake_extract_markdown)

    config = PipelineConfig(
        markdown_dir=tmp_path / "markdown",
        chroma_dir=tmp_path / "chroma",
        manifest_dir=tmp_path / "manifests",
        collection_name="smoke-test",
        embedder_type="hash",
        hash_embedding_dimensions=64,
        chunk_size=120,
        chunk_overlap=20,
    )

    result = ingest_pdf(pdf_path, config)

    assert result.page_count == 2
    assert result.chunk_count >= 1
    assert Path(result.markdown_path).exists()
    assert Path(result.manifest_path).exists()
