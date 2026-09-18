"""Provider-agnostic LLM and embedding interfaces.

Everything downstream (rag/qa.py, app.py) talks only to the LLMProvider and
EmbeddingProvider abstract base classes below. Swapping providers — Anthropic,
any OpenAI-compatible endpoint (OpenAI, Groq, Google AI Studio, Ollama), or a
local Hugging Face embedding model — is a .env change only; see
get_llm_provider() / get_embedding_provider().
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderError(Exception):
    """Raised for any provider failure (auth, rate limit, network, bad
    config). The message is meant to be shown to the end user as-is."""


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's answer as plain text."""


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, same order."""


class AnthropicLLM(LLMProvider):
    def __init__(self, api_key: str, model: str, max_tokens: int = 1024):
        if not api_key:
            raise ProviderError(
                "LLM_API_KEY is not set. Add your Anthropic API key to .env."
            )
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return "".join(
                block.text for block in response.content if block.type == "text"
            )
        except self._anthropic.AuthenticationError as exc:
            raise ProviderError(
                "Anthropic rejected the API key. Check LLM_API_KEY in .env."
            ) from exc
        except self._anthropic.RateLimitError as exc:
            raise ProviderError(
                "Anthropic rate limit hit. Wait a moment and try again."
            ) from exc
        except self._anthropic.APIConnectionError as exc:
            raise ProviderError(
                "Could not reach Anthropic's API. Check your network connection."
            ) from exc
        except self._anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error: {exc.message}") from exc


class OpenAICompatibleLLM(LLMProvider):
    """Works with OpenAI, Groq, Google AI Studio, Ollama, or any other
    OpenAI-compatible chat completions endpoint — just point LLM_BASE_URL
    at it."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        max_tokens: int = 1024,
    ):
        if not api_key:
            raise ProviderError(
                "LLM_API_KEY is not set. Add an API key to .env "
                "(for Ollama, any non-empty placeholder value works)."
            )
        import openai

        self._openai = openai
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except self._openai.AuthenticationError as exc:
            raise ProviderError(
                "The LLM endpoint rejected the API key. Check LLM_API_KEY in .env."
            ) from exc
        except self._openai.RateLimitError as exc:
            raise ProviderError(
                "LLM rate limit hit. Wait a moment and try again."
            ) from exc
        except self._openai.APIConnectionError as exc:
            raise ProviderError(
                "Could not reach the LLM endpoint. Check LLM_BASE_URL and your "
                "network connection (e.g. is Ollama running?)."
            ) from exc
        except self._openai.APIStatusError as exc:
            raise ProviderError(f"LLM API error: {exc.message}") from exc


class OpenAICompatibleEmbedding(EmbeddingProvider):
    """Embeddings via an OpenAI-compatible endpoint (OpenAI, Google AI
    Studio, or Ollama)."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        if not api_key:
            raise ProviderError(
                "EMBEDDING_API_KEY is not set. Add an API key to .env."
            )
        import openai

        self._openai = openai
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(
                model=self._model, input=texts
            )
            return [item.embedding for item in response.data]
        except self._openai.AuthenticationError as exc:
            raise ProviderError(
                "The embedding endpoint rejected the API key. Check "
                "EMBEDDING_API_KEY in .env."
            ) from exc
        except self._openai.RateLimitError as exc:
            raise ProviderError(
                "Embedding provider rate limit hit. Wait a moment and try again."
            ) from exc
        except self._openai.APIConnectionError as exc:
            raise ProviderError(
                "Could not reach the embedding endpoint. Check EMBEDDING_BASE_URL "
                "and your network connection."
            ) from exc
        except self._openai.APIStatusError as exc:
            raise ProviderError(f"Embedding API error: {exc.message}") from exc


class HuggingFaceEmbedding(EmbeddingProvider):
    """Local sentence-transformers model. No API key, no network calls
    after the model weights are downloaded the first time. Default
    embedding provider so the app runs out of the box with only a Claude key."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ProviderError(
                "sentence-transformers is not installed. Run "
                "'pip install -r requirements.txt'."
            ) from exc
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise ProviderError(
                f"Could not load embedding model '{model_name}'. Check "
                "EMBEDDING_MODEL in .env and your network connection "
                "(first run downloads the model)."
            ) from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, show_progress_bar=False)
        return vectors.tolist()


def get_llm_provider(config) -> LLMProvider:
    provider = config.LLM_PROVIDER
    if provider == "anthropic":
        return AnthropicLLM(
            api_key=config.LLM_API_KEY,
            model=config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
        )
    if provider == "openai":
        return OpenAICompatibleLLM(
            api_key=config.LLM_API_KEY,
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            max_tokens=config.LLM_MAX_TOKENS,
        )
    raise ProviderError(
        f"Unknown LLM_PROVIDER '{provider}'. Supported values: anthropic, openai."
    )


def get_embedding_provider(config) -> EmbeddingProvider:
    provider = config.EMBEDDING_PROVIDER
    if provider == "huggingface":
        return HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
    if provider == "openai":
        return OpenAICompatibleEmbedding(
            api_key=config.EMBEDDING_API_KEY,
            model=config.EMBEDDING_MODEL,
            base_url=config.EMBEDDING_BASE_URL,
        )
    raise ProviderError(
        f"Unknown EMBEDDING_PROVIDER '{provider}'. Supported values: "
        "huggingface, openai."
    )
