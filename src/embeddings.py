"""Embedding backends.

Three interchangeable implementations:

* ``GeminiEmbedder`` - Google ``gemini-embedding-001`` through langchain-google-genai
  (free tier, 3072 dimensions).
* ``OpenAIEmbedder`` - ``text-embedding-3-small`` (alternative provider).
* ``LocalHashEmbedder`` - deterministic signed hashing of words and word bigrams,
  used when no API key is configured so ingestion, retrieval and RAG stay fully
  runnable offline. It is not a learned model; the README says so.

Both expose ``embed_documents`` and ``embed_query`` and report an ``identifier``
that is stored in the index manifest, so a provider change forces a rebuild.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Sequence

from config import Settings, get_settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "has", "have", "how", "i", "in", "is", "it", "my", "of", "on", "or",
    "that", "the", "to", "was", "what", "when", "which", "who", "will", "with",
}


def _tokens(text: str) -> list[str]:
    words = [word for word in _TOKEN_RE.findall(text.lower()) if word not in _STOPWORDS]
    bigrams = [f"{first}_{second}" for first, second in zip(words, words[1:])]
    return words + bigrams


class LocalHashEmbedder:
    """Deterministic signed hashing embedder (no model download, no network)."""

    dim = 512
    identifier = "local-hash-bow-v1/512"

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _tokens(text)
        if not tokens:
            return vector

        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class GeminiEmbedder:
    """Google Gemini embeddings via langchain-google-genai."""

    def __init__(self, settings: Settings) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._client = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.llm_api_key,
        )
        self.identifier = f"gemini/{settings.embedding_model}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._client.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


class OpenAIEmbedder:
    """LangChain OpenAI embeddings (works with OpenAI-compatible endpoints too)."""

    def __init__(self, settings: Settings) -> None:
        from langchain_openai import OpenAIEmbeddings

        self._client = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or None,
        )
        self.identifier = f"openai/{settings.embedding_model}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._client.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


_embedder = None


def get_embedder(settings: Settings | None = None):
    """Return a cached embedder for the configured provider."""
    global _embedder
    settings = settings or get_settings()
    if _embedder is None:
        if settings.resolved_embedding_provider == "gemini":
            _embedder = GeminiEmbedder(settings)
        elif settings.resolved_embedding_provider == "openai":
            _embedder = OpenAIEmbedder(settings)
        else:
            _embedder = LocalHashEmbedder()
    return _embedder


def reset_embedder() -> None:
    """Forget the cached embedder (used by tests)."""
    global _embedder
    _embedder = None


def embed_documents(embedder, texts: Iterable[str]) -> list[list[float]]:
    return embedder.embed_documents(list(texts))
