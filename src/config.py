"""Central configuration.

Every setting comes from an environment variable (loaded from ``.env`` by
python-dotenv). This module resolves paths, keys and observability settings,
reports whether an LLM is configured, enables LangSmith tracing and validates the
configuration. It never prints secret values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# The modules live in ``src/``; ``.env``, ``data/`` and ``chroma_db/`` stay at the
# repository root, one level above this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Real environment variables win over .env values (override=False).
load_dotenv(PROJECT_ROOT / ".env", override=False)

SUPPORTED_LLM_PROVIDERS = {"gemini", "groq", "openai", "openai-compatible"}
SUPPORTED_EMBEDDING_PROVIDERS = {"auto", "gemini", "openai", "local"}

# Defaults chosen to work on free tiers. Override with LLM_MODEL / EMBEDDING_MODEL;
# model availability differs per account and changes over time.
CHAT_MODEL_DEFAULTS = {
    "gemini": "gemini-3.5-flash-lite",
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
}
EMBEDDING_MODEL_DEFAULTS = {
    "gemini": "gemini-embedding-001",
    "openai": "text-embedding-3-small",
}


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _path(name: str, default: Path) -> Path:
    """Resolve an environment path; relative paths are relative to the project root."""
    raw = _env(name)
    if not raw:
        return default
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment flag; empty or unset returns ``default``."""
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _gemini_env_key() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        value = _env(name)
        if value:
            return value
    return ""


def _langsmith_key() -> str:
    return _env("LANGSMITH_API_KEY") or _env("LANGCHAIN_API_KEY")


def _provider_api_key() -> str:
    """Resolve the API key for the configured provider.

    ``LLM_API_KEY`` wins when set; otherwise the provider-specific variable is used
    (``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` for Gemini, ``GROQ_API_KEY`` for Groq,
    ``OPENAI_API_KEY`` for OpenAI).
    """
    explicit = _env("LLM_API_KEY")
    if explicit:
        return explicit
    provider = _env("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        return _gemini_env_key()
    if provider == "groq":
        return _env("GROQ_API_KEY")
    return _env("OPENAI_API_KEY")


def _default_chat_model() -> str:
    explicit = _env("LLM_MODEL")
    if explicit:
        return explicit
    provider = _env("LLM_PROVIDER", "gemini").lower()
    return CHAT_MODEL_DEFAULTS.get(provider, CHAT_MODEL_DEFAULTS["gemini"])


def _default_embedding_model() -> str:
    explicit = _env("EMBEDDING_MODEL")
    if explicit:
        return explicit
    provider = _env("EMBEDDING_PROVIDER", "auto").lower()
    if provider in EMBEDDING_MODEL_DEFAULTS:
        return EMBEDDING_MODEL_DEFAULTS[provider]
    llm_provider = _env("LLM_PROVIDER", "gemini").lower()
    if llm_provider == "gemini" or _gemini_env_key():
        return EMBEDDING_MODEL_DEFAULTS["gemini"]
    return EMBEDDING_MODEL_DEFAULTS["openai"]


@dataclass(frozen=True)
class Settings:
    """Resolved application settings."""

    data_dir: Path = field(default_factory=lambda: _path("DATA_DIR", PROJECT_ROOT / "data"))
    chroma_db_path: Path = field(
        default_factory=lambda: _path("CHROMA_DB_PATH", PROJECT_ROOT / "chroma_db")
    )

    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "gemini").lower())
    llm_model: str = field(default_factory=_default_chat_model)
    llm_api_key: str = field(default_factory=_provider_api_key)
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL"))

    embedding_provider: str = field(
        default_factory=lambda: _env("EMBEDDING_PROVIDER", "auto").lower()
    )
    embedding_model: str = field(default_factory=_default_embedding_model)

    langsmith_tracing: bool = field(
        default_factory=lambda: _env_bool("LANGSMITH_TRACING", bool(_langsmith_key()))
    )
    langsmith_api_key: str = field(default_factory=_langsmith_key)
    langsmith_project: str = field(
        default_factory=lambda: _env("LANGSMITH_PROJECT", "ai-academic-advisor")
    )
    langsmith_endpoint: str = field(default_factory=lambda: _env("LANGSMITH_ENDPOINT"))

    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))
    retrieval_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_K", 4))

    @property
    def courses_csv(self) -> Path:
        return self.data_dir / "courses.csv"

    @property
    def programs_json(self) -> Path:
        return self.data_dir / "programs.json"

    @property
    def regulations_pdf(self) -> Path:
        return self.data_dir / "academic_regulations.pdf"

    @property
    def students_json(self) -> Path:
        return self.data_dir / "students.json"

    @property
    def llm_available(self) -> bool:
        """True when a chat model is configured; otherwise the advisor runs offline."""
        return bool(self.llm_api_key)

    @property
    def resolved_embedding_provider(self) -> str:
        """Embedding backend that will actually be used.

        ``auto`` uses the provider's embeddings when a key exists (Gemini by default),
        otherwise the deterministic local embedder. Both index and query embeddings
        always use the same backend.
        """
        if self.embedding_provider in {"gemini", "openai", "local"}:
            return self.embedding_provider
        # Groq only serves chat completions, so auto resolution looks for a Gemini
        # (or OpenAI) key and otherwise falls back to the local embedder.
        if self.llm_provider == "gemini" and self.llm_api_key:
            return "gemini"
        if _gemini_env_key():
            return "gemini"
        if (self.llm_provider == "openai" and self.llm_api_key) or _env("OPENAI_API_KEY"):
            return "openai"
        return "local"

    @property
    def langsmith_enabled(self) -> bool:
        """True when runs should be traced and uploaded to LangSmith."""
        return bool(self.langsmith_tracing and self.langsmith_api_key)

    def apply_tracing_env(self) -> None:
        """Expose tracing settings to LangChain's tracer through the environment.

        LangChain reads these variables when a run starts, so applying them here
        (after ``.env`` is loaded) is enough - no tracer objects to wire manually.
        """
        os.environ["LANGSMITH_TRACING"] = "true" if self.langsmith_enabled else "false"
        if self.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
        if self.langsmith_project:
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
        if self.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = self.langsmith_endpoint

    def validation_errors(self) -> list[str]:
        """Return human-readable configuration problems (empty list when valid)."""
        errors: list[str] = []
        for label, path in (
            ("Courses CSV", self.courses_csv),
            ("Programs JSON", self.programs_json),
            ("Academic regulations PDF", self.regulations_pdf),
        ):
            if not path.exists():
                errors.append(f"{label} not found at {path}")
        if self.llm_provider not in SUPPORTED_LLM_PROVIDERS:
            errors.append(
                f"Unsupported LLM_PROVIDER '{self.llm_provider}'. "
                f"Use one of: {', '.join(sorted(SUPPORTED_LLM_PROVIDERS))}."
            )
        if self.llm_available and not self.llm_model:
            errors.append("LLM_MODEL must be set when LLM_API_KEY is provided.")
        if self.embedding_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
            errors.append(
                f"Unsupported EMBEDDING_PROVIDER '{self.embedding_provider}'. "
                f"Use one of: {', '.join(sorted(SUPPORTED_EMBEDDING_PROVIDERS))}."
            )
        if self.langsmith_tracing and not self.langsmith_api_key:
            errors.append("LANGSMITH_TRACING=true requires LANGSMITH_API_KEY.")
        if self.resolved_embedding_provider == "gemini" and not self.llm_api_key:
            errors.append("EMBEDDING_PROVIDER=gemini requires GEMINI_API_KEY (or LLM_API_KEY).")
        if self.resolved_embedding_provider == "openai" and not self.llm_api_key:
            errors.append("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY (or LLM_API_KEY).")
        if self.chunk_size <= 0:
            errors.append("CHUNK_SIZE must be positive.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            errors.append("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE.")
        if self.retrieval_k <= 0:
            errors.append("RETRIEVAL_K must be positive.")
        return errors

    def describe(self) -> dict[str, str]:
        """Non-secret summary used by the UI and logs."""
        return {
            "Answer mode": (
                f"LLM ({self.llm_model} via {self.llm_provider})"
                if self.llm_available
                else "offline extractive (no LLM_API_KEY set)"
            ),
            "Embeddings": (
                f"Gemini ({self.embedding_model})"
                if self.resolved_embedding_provider == "gemini"
                else f"OpenAI ({self.embedding_model})"
                if self.resolved_embedding_provider == "openai"
                else "local hashing embedder"
            ),
            "Vector store": str(self.chroma_db_path),
            "Chunking": f"{self.chunk_size} chars / {self.chunk_overlap} overlap",
            "Retrieved chunks": str(self.retrieval_k),
            "Tracing": (
                f"LangSmith project '{self.langsmith_project}'"
                if self.langsmith_enabled
                else "disabled (set LANGSMITH_API_KEY)"
            ),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings object (cached) and enable tracing."""
    settings = Settings()
    settings.apply_tracing_env()
    return settings
