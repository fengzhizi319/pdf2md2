from pathlib import Path

from pdf2md_rag.embeddings import HashingEmbedder
from pdf2md_rag.models import Chunk
from pdf2md_rag.search import search_chunks
from pdf2md_rag.vectorstore import upsert_chunks


def test_search_chunks_returns_rag_ready_structure(tmp_path: Path) -> None:
    embedder = HashingEmbedder(dimensions=64)
    chunks = [
        Chunk(
            chunk_id="chunk-1",
            text="Lasso is a lookup argument protocol for efficient verification.",
            metadata={
                "source_path": "/tmp/lasso.pdf",
                "source_name": "lasso",
                "heading": "Abstract",
                "page": 1,
                "chunk_index": 0,
                "text_length": 62,
            },
        ),
        Chunk(
            chunk_id="chunk-2",
            text="RAG systems store chunks in vector databases like Chroma.",
            metadata={
                "source_path": "/tmp/lasso.pdf",
                "source_name": "lasso",
                "heading": "Implementation",
                "page": 2,
                "chunk_index": 1,
                "text_length": 59,
            },
        ),
    ]
    upsert_chunks(
        chunks=chunks,
        embeddings=embedder.embed_texts([chunk.text for chunk in chunks]),
        persist_directory=tmp_path / "chroma",
        collection_name="search-test",
    )

    result = search_chunks(
        question="What is Lasso?",
        collection_name="search-test",
        persist_directory=tmp_path / "chroma",
        top_k=2,
        embedder_type="hash",
        embedding_model="unused",
        hash_dimensions=64,
        max_context_chars=800,
    )

    assert result.question == "What is Lasso?"
    assert len(result.hits) == 2
    assert result.hits[0].chunk_id
    assert result.hits[0].citation.startswith("lasso")
    assert "Relevant context:" in result.context_text
    assert result.sources
    assert result.retrieval_meta["hit_count"] == 2

