from pathlib import Path

from pdf2md_rag.embeddings import HashingEmbedder
from pdf2md_rag.models import Chunk
from pdf2md_rag.vectorstore import query_collection, upsert_chunks


def test_chroma_persistence_and_query(tmp_path: Path) -> None:
    embedder = HashingEmbedder(dimensions=64)
    chunks = [
        Chunk(chunk_id="a", text="apple orange banana", metadata={"page": 1}),
        Chunk(chunk_id="b", text="zk proofs lasso lookup", metadata={"page": 2}),
    ]
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])

    summary = upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
        persist_directory=tmp_path / "chroma",
        collection_name="test-collection",
    )

    assert summary["count"] == 2

    rows = query_collection(
        question="lookup protocol",
        query_embedding=embedder.embed_query("lookup protocol"),
        persist_directory=tmp_path / "chroma",
        collection_name="test-collection",
        top_k=1,
    )

    assert len(rows) == 1
    assert rows[0]["metadata"]["page"] in {1, 2}

