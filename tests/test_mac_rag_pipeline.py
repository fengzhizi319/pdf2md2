import sys
import types
from pathlib import Path

from mac_rag_pipeline import chunk_markdown_with_math_protection, parse_pdf_to_md


class _FakeRendered:
    def __init__(self, markdown: str):
        self.markdown = markdown


def test_mac_rag_pipeline_uses_marker_and_chunks(monkeypatch) -> None:
    project_pdf_dir = Path(__file__).resolve().parents[1] / "pdf"
    pdf_path = project_pdf_dir / "Understanding Lasso – A Novel Lookup Argument Protocol.pdf"
    output_md = project_pdf_dir / "math-demo.marker.md"

    marker_module = types.ModuleType("marker")
    converters_module = types.ModuleType("marker.converters")
    pdf_module = types.ModuleType("marker.converters.pdf")
    calls: dict[str, object] = {}

    class FakePdfConverter:
        default_processors = [
            type("TableProcessor", (), {"__module__": "marker.processors.table", "__name__": "TableProcessor"}),
            type("TextProcessor", (), {"__module__": "marker.processors.text", "__name__": "TextProcessor"}),
        ]

        def __init__(self, artifact_dict, config=None, processor_list=None, **kwargs):
            calls["artifact_dict"] = artifact_dict
            calls["config"] = config or {}
            calls["processor_list"] = processor_list
            calls["kwargs"] = kwargs
            self.page_count = 2

        def __call__(self, filepath: str):
            calls["filepath"] = filepath
            return _FakeRendered("# Title\n\nEquation block:\n\n$$a^2+b^2=c^2$$\n")

    pdf_module.PdfConverter = FakePdfConverter
    marker_module.converters = converters_module
    converters_module.pdf = pdf_module

    monkeypatch.setitem(sys.modules, "marker", marker_module)
    monkeypatch.setitem(sys.modules, "marker.converters", converters_module)
    monkeypatch.setitem(sys.modules, "marker.converters.pdf", pdf_module)
    monkeypatch.setattr(
        "pdf2md_rag.pdf_to_markdown.load_marker_models",
        lambda device, disable_table_rec=False: {"device": device, "disable_table_rec": disable_table_rec},
    )

    if output_md.exists():
        output_md.unlink()

    try:
        markdown_text = parse_pdf_to_md(pdf_path, output_md, device="mps")
        chunks = chunk_markdown_with_math_protection(markdown_text)

        assert output_md.exists()
        assert "$$a^2+b^2=c^2$$" in markdown_text
        assert len(chunks) >= 1
        assert any("$$a^2+b^2=c^2$$" in chunk for chunk in chunks)
        assert calls["filepath"] == str(pdf_path.resolve())
        assert calls["artifact_dict"] == {"device": "mps", "disable_table_rec": True}
        assert calls["config"] == {"disable_multiprocessing": True, "output_format": "markdown"}
        assert calls["processor_list"] == ["marker.processors.text.TextProcessor"]
    finally:
        if output_md.exists():
            output_md.unlink()


def test_chunk_markdown_with_math_protection_preserves_formula_boundaries() -> None:
    markdown_text = "# Title\n\nIntro paragraph.\n\n$$a^2+b^2=c^2$$\n\nConclusion."

    chunks = chunk_markdown_with_math_protection(markdown_text)

    assert len(chunks) >= 1
    assert any("$$a^2+b^2=c^2$$" in chunk for chunk in chunks)
