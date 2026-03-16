import sys
import types
from pathlib import Path

from pdf2md_rag.pdf_to_markdown import extract_markdown


class _FakeRendered:
    def __init__(self, markdown: str):
        self.markdown = markdown


def test_extract_markdown_disables_table_rec_on_mps(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")

    marker_module = types.ModuleType("marker")
    converters_module = types.ModuleType("marker.converters")
    pdf_module = types.ModuleType("marker.converters.pdf")
    calls: dict[str, object] = {}

    class FakePdfConverter:
        default_processors = [
            type("TableProcessor", (), {"__module__": "marker.processors.table", "__name__": "TableProcessor"}),
            type("LLMTableProcessor", (), {"__module__": "marker.processors.llm.llm_table", "__name__": "LLMTableProcessor"}),
            type("TextProcessor", (), {"__module__": "marker.processors.text", "__name__": "TextProcessor"}),
        ]

        def __init__(self, artifact_dict, config=None, processor_list=None, **kwargs):
            calls["artifact_dict"] = artifact_dict
            calls["config"] = config or {}
            calls["processor_list"] = processor_list
            self.page_count = 3

        def __call__(self, filepath: str):
            calls["filepath"] = filepath
            return _FakeRendered("# Marker Title\n\n$$x^2+y^2=z^2$$")

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

    document = extract_markdown(pdf_path, device="mps")

    assert document.source_path == pdf_path.resolve()
    assert document.page_count == 3
    assert "$$x^2+y^2=z^2$$" in document.text
    assert calls["filepath"] == str(pdf_path.resolve())
    assert calls["artifact_dict"] == {"device": "mps", "disable_table_rec": True}
    assert calls["config"] == {"disable_multiprocessing": True, "output_format": "markdown"}
    assert calls["processor_list"] == ["marker.processors.text.TextProcessor"]
