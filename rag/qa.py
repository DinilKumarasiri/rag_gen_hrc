"""Orchestrates the two end-to-end flows on top of ingest/chunking/store/
providers: indexing an uploaded file, and answering a question against an
indexed collection."""
from __future__ import annotations

from rag.chunking import chunk_text
from rag.ingest import extract_text
from rag.providers import EmbeddingProvider, LLMProvider
from rag.store import VectorStore

NO_MATCH_ANSWER = "I couldn't find that in the provided documents."

# Chroma collections are created with hnsw:space="cosine", so distance here
# is 1 - cosine_similarity (0 = identical, 2 = opposite). Chunks scoring
# above this are treated as unrelated noise rather than real matches, so a
# question with no relevant chunks gets NO_MATCH_ANSWER instead of the LLM
# improvising over irrelevant context. Tuned empirically; see README.
DISTANCE_THRESHOLD = 0.5

SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the user's question "
    "using ONLY the numbered context excerpts below. Do not use outside "
    "knowledge and do not guess. If the excerpts don't contain enough "
    "information to answer, say so explicitly. Refer to excerpts you used "
    "by their number, e.g. [1]."
)


def ingest_file(
    filename: str,
    file_bytes: bytes,
    collection_name: str,
    store: VectorStore,
    embedder: EmbeddingProvider,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """Extract, chunk, embed, and store one uploaded file.

    Re-uploading the same filename to the same collection replaces its
    previously indexed chunks (see VectorStore.add_chunks). Returns the
    number of chunks indexed.
    """
    text = extract_text(filename, file_bytes)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    embeddings = embedder.embed(chunks)
    store.add_chunks(collection_name, filename, chunks, embeddings)
    return len(chunks)


def answer_question(
    question: str,
    collection_name: str,
    store: VectorStore,
    embedder: EmbeddingProvider,
    llm: LLMProvider,
    top_k: int,
) -> dict:
    """Retrieve relevant chunks and generate an answer grounded only in
    them. Returns {"answer": str, "citations": [{source, chunk_index,
    snippet}]}. Falls back to NO_MATCH_ANSWER — without calling the LLM at
    all — when nothing relevant is found."""
    query_embedding = embedder.embed([question])[0]
    hits = store.query(collection_name, query_embedding, top_k)
    relevant_hits = [hit for hit in hits if hit["distance"] <= DISTANCE_THRESHOLD]

    if not relevant_hits:
        return {"answer": NO_MATCH_ANSWER, "citations": []}

    context_block = "\n\n".join(
        f"[{i + 1}] (source: {hit['source']}, chunk {hit['chunk_index']})\n{hit['text']}"
        for i, hit in enumerate(relevant_hits)
    )
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"
    answer = llm.generate(SYSTEM_PROMPT, user_prompt)

    citations = [
        {
            "source": hit["source"],
            "chunk_index": hit["chunk_index"],
            "snippet": hit["text"][:300],
        }
        for hit in relevant_hits
    ]
    return {"answer": answer, "citations": citations}
