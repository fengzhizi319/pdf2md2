from pathlib import Path

from typer.testing import CliRunner

from pdf2md_rag.cli import app
from pdf2md_rag.pipeline import IngestResult

runner = CliRunner()


def test_ingest_cli_verifies_output_files(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")
    markdown_path = tmp_path / "demo.md"
    manifest_path = tmp_path / "demo.json"
    markdown_path.write_text("# demo", encoding="utf-8")
    manifest_path.write_text("{}", encoding="utf-8")

    def fake_ingest_pdf(pdf_path: Path, config) -> IngestResult:
        return IngestResult(
            pdf_path=str(pdf_path),
            markdown_path=str(markdown_path),
            manifest_path=str(manifest_path),
            collection_name=config.collection_name,
            chunk_count=1,
            vector_count=1,
            page_count=1,
            embedder_type=config.embedder_type,
            embedding_model=config.embedding_model,
        )

    monkeypatch.setattr("pdf2md_rag.cli.ingest_pdf", fake_ingest_pdf)

    result = runner.invoke(app, ["ingest", str(pdf_path), "--embedder", "hash"])

    assert result.exit_code == 0
    assert "Verified Markdown:" in result.stdout
    assert "Verified Manifest:" in result.stdout
    assert str(markdown_path.resolve()) in result.stdout


def test_ingest_cli_fails_when_output_file_is_missing(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")
    missing_markdown_path = tmp_path / "missing.md"
    manifest_path = tmp_path / "demo.json"
    manifest_path.write_text("{}", encoding="utf-8")

    def fake_ingest_pdf(pdf_path: Path, config) -> IngestResult:
        return IngestResult(
            pdf_path=str(pdf_path),
            markdown_path=str(missing_markdown_path),
            manifest_path=str(manifest_path),
            collection_name=config.collection_name,
            chunk_count=1,
            vector_count=1,
            page_count=1,
            embedder_type=config.embedder_type,
            embedding_model=config.embedding_model,
        )

    monkeypatch.setattr("pdf2md_rag.cli.ingest_pdf", fake_ingest_pdf)

    result = runner.invoke(app, ["ingest", str(pdf_path), "--embedder", "hash"])

    assert result.exit_code == 1
    assert "ERROR: Markdown file was not created:" in result.stdout

