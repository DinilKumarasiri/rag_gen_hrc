import pytest

from rag.store import DimensionMismatchError, VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "chroma_db"))


def test_query_on_empty_collection_returns_no_hits(store):
    store.create_collection("docs")
    assert store.query("docs", query_embedding=[0.1, 0.2, 0.3], top_k=4) == []


def test_add_and_query_returns_closest_chunk_first(store):
    store.add_chunks(
        "docs",
        source_filename="a.txt",
        chunks=["chunk about cats", "chunk about dogs"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    hits = store.query("docs", query_embedding=[0.9, 0.1, 0.0], top_k=1)
    assert len(hits) == 1
    assert hits[0]["text"] == "chunk about cats"
    assert hits[0]["source"] == "a.txt"
    assert hits[0]["chunk_index"] == 0


def test_query_respects_top_k(store):
    store.add_chunks(
        "docs",
        source_filename="a.txt",
        chunks=["one", "two", "three"],
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )
    hits = store.query("docs", query_embedding=[1.0, 0.0], top_k=2)
    assert len(hits) == 2


def test_reuploading_same_filename_replaces_old_chunks(store):
    store.add_chunks(
        "docs",
        source_filename="a.txt",
        chunks=["old chunk one", "old chunk two", "old chunk three"],
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
    )
    assert store.collection_count("docs") == 3

    store.add_chunks(
        "docs",
        source_filename="a.txt",
        chunks=["new chunk one"],
        embeddings=[[1.0, 0.0]],
    )
    assert store.collection_count("docs") == 1
    hits = store.query("docs", query_embedding=[1.0, 0.0], top_k=5)
    assert [h["text"] for h in hits] == ["new chunk one"]


def test_reuploading_does_not_affect_other_files_chunks(store):
    store.add_chunks(
        "docs", source_filename="a.txt", chunks=["a chunk"], embeddings=[[1.0, 0.0]]
    )
    store.add_chunks(
        "docs", source_filename="b.txt", chunks=["b chunk"], embeddings=[[0.0, 1.0]]
    )
    store.add_chunks(
        "docs",
        source_filename="a.txt",
        chunks=["a chunk updated"],
        embeddings=[[1.0, 0.0]],
    )
    assert store.collection_count("docs") == 2


def test_dimension_mismatch_on_add_raises_clear_error(store):
    store.add_chunks(
        "docs", source_filename="a.txt", chunks=["chunk"], embeddings=[[1.0, 0.0, 0.0]]
    )
    with pytest.raises(DimensionMismatchError):
        store.add_chunks(
            "docs",
            source_filename="b.txt",
            chunks=["other chunk"],
            embeddings=[[1.0, 0.0]],
        )


def test_dimension_mismatch_on_query_raises_clear_error(store):
    store.add_chunks(
        "docs", source_filename="a.txt", chunks=["chunk"], embeddings=[[1.0, 0.0, 0.0]]
    )
    with pytest.raises(DimensionMismatchError):
        store.query("docs", query_embedding=[1.0, 0.0], top_k=1)


def test_list_collections(store):
    store.create_collection("alpha")
    store.create_collection("beta")
    assert store.list_collections() == ["alpha", "beta"]
