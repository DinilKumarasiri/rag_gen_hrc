import pytest

from rag.qa import NO_MATCH_ANSWER, answer_question, ingest_file
from rag.store import VectorStore


class FakeEmbeddingProvider:
    """Deterministic stand-in: embeds by returning a fixed vector per exact
    text, so tests don't need a real model or network access."""

    def __init__(self, vectors_by_text: dict):
        self._vectors_by_text = vectors_by_text

    def embed(self, texts):
        return [self._vectors_by_text[text] for text in texts]


class FakeLLMProvider:
    def __init__(self):
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return "The PTO carryover cap is 5 days. [1]"


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "chroma_db"))


def test_answer_question_on_empty_collection_never_calls_llm(store):
    embedder = FakeEmbeddingProvider({"what is the pto policy?": [1.0, 0.0]})
    llm = FakeLLMProvider()

    result = answer_question(
        "what is the pto policy?", "docs", store, embedder, llm, top_k=4
    )

    assert result["answer"] == NO_MATCH_ANSWER
    assert result["citations"] == []
    assert llm.calls == []


def test_answer_question_below_threshold_never_calls_llm(store):
    store.add_chunks(
        "docs",
        source_filename="unrelated.txt",
        chunks=["something about kitchen recipes"],
        embeddings=[[0.0, 1.0]],
    )
    embedder = FakeEmbeddingProvider({"what is the pto policy?": [1.0, 0.0]})
    llm = FakeLLMProvider()

    result = answer_question(
        "what is the pto policy?", "docs", store, embedder, llm, top_k=4
    )

    assert result["answer"] == NO_MATCH_ANSWER
    assert result["citations"] == []
    assert llm.calls == []


def test_answer_question_with_relevant_hit_calls_llm_and_returns_citations(store):
    store.add_chunks(
        "docs",
        source_filename="pto_policy.md",
        chunks=["Up to 5 unused PTO days may be carried over."],
        embeddings=[[1.0, 0.0]],
    )
    embedder = FakeEmbeddingProvider({"what is the pto carryover cap?": [1.0, 0.0]})
    llm = FakeLLMProvider()

    result = answer_question(
        "what is the pto carryover cap?", "docs", store, embedder, llm, top_k=4
    )

    assert "5 days" in result["answer"]
    assert len(llm.calls) == 1
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source"] == "pto_policy.md"
    assert result["citations"][0]["chunk_index"] == 0
    assert "carried over" in result["citations"][0]["snippet"]


def test_ingest_file_chunks_embeds_and_stores(store):
    text = "one two three four five six seven eight"
    embedder = FakeEmbeddingProvider(
        {"one two three four": [1.0, 0.0], "five six seven eight": [0.0, 1.0]}
    )

    chunk_count = ingest_file(
        "notes.txt",
        text.encode("utf-8"),
        "docs",
        store,
        embedder,
        chunk_size=4,
        chunk_overlap=0,
    )

    assert chunk_count == 2
    assert store.collection_count("docs") == 2
