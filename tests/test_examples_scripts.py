import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest


def test_example_scripts_are_syntax_valid() -> None:
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    example_files = sorted(path for path in examples_dir.glob("*.py") if path.name != "__init__.py")

    expected = {
        "_debug_common.py",
        "debug_chunking.py",
        "debug_embeddings.py",
        "debug_extract_markdown.py",
        "debug_ingest.py",
        "debug_langgraph.py",
        "debug_qa.py",
        "debug_search.py",
        "debug_vectorstore.py",
    }
    assert expected.issubset({path.name for path in example_files})

    for file_path in example_files:
        py_compile.compile(str(file_path), doraise=True)


def test_embedding_example_scripts_run_successfully_when_enabled() -> None:
    if os.getenv("PDF2MD_RAG_RUN_EMBEDDING_EXAMPLES") != "1":
        pytest.skip("Set PDF2MD_RAG_RUN_EMBEDDING_EXAMPLES=1 to run example integration tests.")

    project_root = Path(__file__).resolve().parents[1]
    scripts = [
        "examples/debug_embeddings.py",
        "examples/debug_vectorstore.py",
        "examples/debug_search.py",
        "examples/debug_qa.py",
    ]

    for script in scripts:
        completed = subprocess.run(
            [sys.executable, script],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
        assert "===" in completed.stdout
