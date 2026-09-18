"""Loads every tunable setting from the environment. Nothing else in this
codebase should call os.getenv() directly — this is the single source of
truth for configuration."""
import os

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


class Config:
    # LLM
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
    LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-5")
    LLM_MAX_TOKENS = _int_env("LLM_MAX_TOKENS", 1024)

    # Embeddings
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL") or None

    # Chunking / retrieval
    CHUNK_SIZE = _int_env("CHUNK_SIZE", 500)
    CHUNK_OVERLAP = _int_env("CHUNK_OVERLAP", 50)
    TOP_K = _int_env("TOP_K", 4)

    # Storage
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
