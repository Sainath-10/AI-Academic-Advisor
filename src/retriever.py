"""Semantic retrieval over the persisted Chroma collections.

A query is embedded with the same embedder used at index time, Chroma returns the
nearest chunks (optionally filtered by metadata), and the results are re-ranked
with a small lexical component.

Why the hybrid score? Vector search supplies the semantic signal. The lexical
overlap keeps exact academic terms ("probation", "residency", course codes) from
being missed, which matters when the offline hashing embedder is in use.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from config import Settings, get_settings
from embeddings import get_embedder
from vector_store import COURSES_COLLECTION, DOCUMENTS_COLLECTION, get_collection

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class RetrievedChunk:
    """One regulation passage with its citation metadata."""

    text: str
    source: str
    page: int | None = None
    section: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        page = f"p. {self.page}" if self.page is not None else "page unknown"
        return f"{self.source}, {page}" + (f" - {self.section}" if self.section else "")

    def to_source(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "label": self.label,
            "kind": "document",
            "text": self.text,
            "score": round(self.score, 4),
        }


@dataclass
class RetrievedCourse:
    """One course catalogue entry returned by vector search."""

    course_code: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _terms(text: str) -> list[str]:
    return [word for word in _WORD_RE.findall((text or "").lower()) if len(word) > 1]


def _lexical_overlap(query: str, text: str) -> float:
    """Query-term coverage plus saturating term frequency of a passage."""
    query_terms = set(_terms(query))
    if not query_terms:
        return 0.0
    passage = Counter(_terms(text))
    matched = [term for term in query_terms if passage[term]]
    if not matched:
        return 0.0
    coverage = len(matched) / len(query_terms)
    hits = sum(passage[term] for term in matched)
    return 0.65 * coverage + 0.35 * (hits / (hits + 4))


def _normalise(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high - low < 1e-9:
        return [1.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def _hybrid_rank(query: str, candidates: list[tuple], semantic_weight: float = 0.6) -> list[tuple]:
    """Combine normalised vector similarity with lexical overlap."""
    if not candidates:
        return []
    semantic_scores = _normalise([candidate[2] for candidate in candidates])
    ranked = []
    for (text, metadata, _distance), semantic in zip(candidates, semantic_scores):
        lexical = _lexical_overlap(query, text)
        combined = semantic_weight * semantic + (1 - semantic_weight) * lexical
        ranked.append((text, metadata, combined, semantic, lexical))
    ranked.sort(key=lambda item: -item[2])
    return ranked


def _build_where(doc_type: str | None, source: str | None) -> dict[str, Any] | None:
    clauses = []
    if doc_type:
        clauses.append({"doc_type": {"$eq": doc_type}})
    if source:
        clauses.append({"source": {"$eq": source}})
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses} if clauses else None


def _query(collection, query: str, n_results: int, where, settings: Settings) -> list[tuple]:
    total = collection.count()
    if total == 0:
        return []
    result = collection.query(
        query_embeddings=[get_embedder(settings).embed_query(query)],
        n_results=max(1, min(n_results, total)),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    return [
        (text or "", metadata or {}, 1.0 - float(distance))
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]


def search_documents(
    query: str,
    k: int = 4,
    doc_type: str | None = None,
    source: str | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant regulation chunks, with page metadata."""
    settings = settings or get_settings()
    try:
        collection = get_collection(DOCUMENTS_COLLECTION, settings)
    except Exception:
        return []

    candidates = _query(collection, query, max(12, k * 4), _build_where(doc_type, source), settings)

    chunks: list[RetrievedChunk] = []
    for text, metadata, combined, semantic, lexical in _hybrid_rank(query, candidates)[:k]:
        chunks.append(
            RetrievedChunk(
                text=text,
                source=str(metadata.get("source", "unknown")),
                page=int(metadata["page"]) if metadata.get("page") is not None else None,
                section=str(metadata.get("section", "")),
                score=combined,
                metadata={**dict(metadata), "semantic_score": round(semantic, 4), "lexical_score": round(lexical, 4)},
            )
        )
    return chunks


def search_course_documents(
    query: str,
    k: int = 5,
    department: str | None = None,
    min_level: int | None = None,
    settings: Settings | None = None,
) -> list[RetrievedCourse]:
    """Retrieve course catalogue entries with optional metadata filters."""
    settings = settings or get_settings()
    clauses = []
    if department:
        clauses.append({"department": {"$eq": department}})
    if min_level:
        clauses.append({"level": {"$gte": int(min_level)}})
    where = None
    if len(clauses) == 1:
        where = clauses[0]
    elif clauses:
        where = {"$and": clauses}

    try:
        collection = get_collection(COURSES_COLLECTION, settings)
    except Exception:
        return []

    candidates = _query(collection, query, max(20, k * 4), where, settings)

    courses: list[RetrievedCourse] = []
    for text, metadata, combined, _semantic, _lexical in _hybrid_rank(query, candidates)[:k]:
        courses.append(
            RetrievedCourse(
                course_code=str(metadata.get("course_code", "")),
                text=text,
                score=combined,
                metadata=dict(metadata),
            )
        )
    return courses
