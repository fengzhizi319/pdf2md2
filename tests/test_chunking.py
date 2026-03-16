from pathlib import Path

from pdf2md_rag.chunking import chunk_markdown
from pdf2md_rag.models import MarkdownDocument


def test_chunk_markdown_preserves_metadata_and_splits_large_text() -> None:
    document = MarkdownDocument(
        source_path=Path("sample.pdf"),
        page_count=2,
        text=(
            "# Title\n\n"
            "## Page 1\n\n"
            + ("Alpha beta gamma delta. " * 40)
            + "\n\n## Page 2\n\n"
            + ("Zeta eta theta iota. " * 40)
        ),
    )

    chunks = chunk_markdown(document, chunk_size=240, chunk_overlap=40)

    assert len(chunks) >= 3
    assert all(chunk.metadata["source_name"] == "sample" for chunk in chunks)
    assert chunks[0].metadata["chunk_index"] == 0
    assert any(chunk.metadata["page"] == 2 for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_chunk_markdown_rejects_invalid_overlap() -> None:
    document = MarkdownDocument(source_path=Path("sample.pdf"), page_count=1, text="hello")

    try:
        chunk_markdown(document, chunk_size=100, chunk_overlap=100)
    except ValueError as exc:
        assert "smaller than chunk_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid overlap")

