"""Thin wrapper around a persistent ChromaDB client.

We always supply our own embeddings (from an EmbeddingProvider) rather than
using Chroma's built-in embedding functions, so the embedding provider stays
swappable via config alone.
"""
from __future__ import annotations

import chromadb
from chromadb.errors import InvalidDimensionException


class DimensionMismatchError(Exception):
    """Raised when the current EMBEDDING_MODEL's vector size doesn't match
    a collection that was indexed with a different model."""


class VectorStore:
    def __init__(self, persist_dir: str):
        self._client = chromadb.PersistentClient(path=persist_dir)

    def list_collections(self) -> list[str]:
        return sorted(c.name for c in self._client.list_collections())

    def create_collection(self, name: str) -> None:
        self._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def _get_collection(self, name: str):
        return self._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def collection_count(self, name: str) -> int:
        return self._get_collection(name).count()

    def delete_document(self, collection_name: str, source_filename: str) -> None:
        """Remove all chunks previously indexed for this filename, so
        re-uploading the same file replaces rather than duplicates it."""
        collection = self._get_collection(collection_name)
        collection.delete(where={"source": source_filename})

    def add_chunks(
        self,
        collection_name: str,
        source_filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        collection = self._get_collection(collection_name)

        # Re-uploading the same filename replaces its old chunks instead of
        # duplicating them.
        self.delete_document(collection_name, source_filename)

        ids = [f"{source_filename}::{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": source_filename, "chunk_index": i} for i in range(len(chunks))
        ]
        try:
            collection.add(
                ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
            )
        except InvalidDimensionException as exc:
            raise DimensionMismatchError(
                f"'{source_filename}' produced embeddings with a different "
                f"dimension than what collection '{collection_name}' was "
                "indexed with. This usually means EMBEDDING_MODEL was changed "
                "after this collection was created. Delete the collection "
                "(or the CHROMA_PERSIST_DIR data for it) and re-index all "
                "documents with the current EMBEDDING_MODEL."
            ) from exc

    def query(
        self, collection_name: str, query_embedding: list[float], top_k: int
    ) -> list[dict]:
        collection = self._get_collection(collection_name)
        count = collection.count()
        if count == 0:
            return []
        try:
            results = collection.query(
                query_embeddings=[query_embedding], n_results=min(top_k, count)
            )
        except InvalidDimensionException as exc:
            raise DimensionMismatchError(
                f"The current EMBEDDING_MODEL produces a different vector "
                f"dimension than collection '{collection_name}' was indexed "
                "with. Delete the collection and re-index with the current "
                "EMBEDDING_MODEL."
            ) from exc

        hits = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        for document, metadata, distance in zip(documents, metadatas, distances):
            hits.append(
                {
                    "text": document,
                    "source": metadata["source"],
                    "chunk_index": metadata["chunk_index"],
                    "distance": distance,
                }
            )
        return hits
