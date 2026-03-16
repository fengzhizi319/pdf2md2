from pathlib import Path

from pdf2md_rag.embeddings import HashingEmbedder
from pdf2md_rag.models import Chunk
from pdf2md_rag.simple_qa import ask_question
from pdf2md_rag.vectorstore import upsert_chunks


def _build_store(tmp_path: Path) -> Path:
    embedder = HashingEmbedder(dimensions=64)
    chunks = [
        Chunk(
            chunk_id="chunk-1",
            text="Lasso is a lookup argument protocol used in proof systems.",
            metadata={
                "source_path": "/tmp/lasso.pdf",
                "source_name": "lasso",
                "heading": "Overview",
                "page": 1,
                "chunk_index": 0,
                "text_length": 60,
            },
        )
    ]
    persist_directory = tmp_path / "chroma"
    upsert_chunks(
        chunks=chunks,
        embeddings=embedder.embed_texts([chunk.text for chunk in chunks]),
        persist_directory=persist_directory,
        collection_name="qa-test",
    )
    return persist_directory


def test_ask_question_with_openai_compatible_provider(tmp_path: Path, monkeypatch) -> None:
    persist_directory = _build_store(tmp_path)

    def fake_post_json(url: str, payload: dict, headers: dict) -> dict:
        assert url.endswith("/v1/chat/completions")
        assert payload["messages"][1]["content"].find("Context:") != -1
        return {
            "choices": [
                {
                    "message": {
                        "content": "Lasso is a lookup argument protocol. Sources: [Source 1]"
                    }
                }
            ]
        }

    monkeypatch.setattr("pdf2md_rag.simple_qa._post_json", fake_post_json)

    result = ask_question(
        question="What is Lasso?",
        collection_name="qa-test",
        chroma_dir=persist_directory,
        top_k=1,
        embedder_type="hash",
        embedding_model="unused",
        hash_dimensions=64,
        llm_provider="openai-compatible",
        llm_model="fake-model",
        llm_base_url="http://localhost:8000",
    )

    assert "Lasso is a lookup argument protocol" in result.answer
    assert result.search_result.sources


def test_ask_question_with_ollama_provider(tmp_path: Path, monkeypatch) -> None:
    persist_directory = _build_store(tmp_path)

    def fake_post_json(url: str, payload: dict, headers: dict) -> dict:
        assert url.endswith("/api/chat")
        assert payload["model"] == "llama3.1"
        return {"message": {"content": "It is a lookup protocol. Sources: [Source 1]"}}

    monkeypatch.setattr("pdf2md_rag.simple_qa._post_json", fake_post_json)

    result = ask_question(
        question="What is Lasso?",
        collection_name="qa-test",
        chroma_dir=persist_directory,
        top_k=1,
        embedder_type="hash",
        embedding_model="unused",
        hash_dimensions=64,
        llm_provider="ollama",
        llm_model="llama3.1",
        llm_base_url="http://localhost:11434",
    )

    assert "lookup protocol" in result.answer
    assert result.provider == "ollama"
