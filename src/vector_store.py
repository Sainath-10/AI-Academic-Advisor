"""Persistent ChromaDB storage.

Two collections live in ``CHROMA_DB_PATH``:

* ``academic_documents`` - chunks of the academic regulations PDF.
* ``course_catalog``    - one document per course, used for semantic course search.

Embeddings are computed by ``embeddings.py`` and passed to Chroma explicitly, so the
same vectors are used for indexing and for querying. A manifest file stores a
signature of the ingested data plus collection counts, which lets the application
skip re-indexing on every startup.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import Settings, get_settings

DOCUMENTS_COLLECTION = "academic_documents"
COURSES_COLLECTION = "course_catalog"
MANIFEST_FILE = "index_manifest.json"


@lru_cache(maxsize=4)
def _client_for(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=path,
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )


def get_client(settings: Settings | None = None):
    """Return the persistent Chroma client for the configured directory."""
    settings = settings or get_settings()
    return _client_for(str(settings.chroma_db_path))


def get_collection(name: str, settings: Settings | None = None):
    """Open (or create) a cosine-similarity collection."""
    return get_client(settings).get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def reset_collections(settings: Settings | None = None) -> None:
    """Delete both collections so an index can be rebuilt without duplicates."""
    client = get_client(settings)
    for name in (DOCUMENTS_COLLECTION, COURSES_COLLECTION):
        try:
            client.delete_collection(name)
        except Exception:  # collection does not exist yet
            pass


def add_records(
    collection_name: str,
    records: list[dict[str, Any]],
    embeddings: list[list[float]],
    settings: Settings | None = None,
) -> int:
    """Insert pre-computed records (``id``, ``text``, ``metadata``) with embeddings."""
    if not records:
        return 0
    collection = get_collection(collection_name, settings)
    collection.add(
        ids=[record["id"] for record in records],
        documents=[record["text"] for record in records],
        metadatas=[record["metadata"] for record in records],
        embeddings=embeddings,
    )
    return len(records)


def count(collection_name: str, settings: Settings | None = None) -> int:
    """Number of stored records, or 0 when the collection does not exist yet."""
    try:
        return get_collection(collection_name, settings).count()
    except Exception:
        return 0


def collection_counts(settings: Settings | None = None) -> dict[str, int]:
    return {
        "documents": count(DOCUMENTS_COLLECTION, settings),
        "courses": count(COURSES_COLLECTION, settings),
    }


def manifest_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.chroma_db_path / MANIFEST_FILE


def read_manifest(settings: Settings | None = None) -> dict[str, Any]:
    path = manifest_path(settings)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_manifest(payload: dict[str, Any], settings: Settings | None = None) -> None:
    path = manifest_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
