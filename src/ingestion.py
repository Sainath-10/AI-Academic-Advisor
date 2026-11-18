"""Data ingestion.

Responsibilities:

* read the academic regulations PDF page by page (PyMuPDF),
* read the structured course catalogue (Pandas/CSV),
* read degree requirements and sample student profiles (JSON),
* chunk the PDF pages and embed everything into ChromaDB,
* skip re-indexing when the data, chunking settings and embedder are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import pymupdf

from chunking import split_page
from config import Settings, get_settings
from embeddings import get_embedder
from vector_store import (
    COURSES_COLLECTION,
    DOCUMENTS_COLLECTION,
    add_records,
    collection_counts,
    read_manifest,
    reset_collections,
    write_manifest,
)

_CODE_RE = re.compile(r"^\s*([A-Za-z]{2,6})[\s-]*(\d{2,4}[A-Za-z]?)\s*$")
_CODE_TOKEN_RE = re.compile(r"\b([A-Za-z]{2,4})[\s-]?(\d{3})\b")


# ------------------------------------------------------------------ course codes
def normalize_course_code(value: str) -> str:
    """Normalise ``cs201`` / ``CS-201`` to the catalogue form ``CS 201``."""
    match = _CODE_RE.match(value or "")
    if not match:
        return (value or "").strip().upper()
    return f"{match.group(1).upper()} {match.group(2).upper()}"


def extract_course_codes(text: str, known_codes: list[str] | None = None) -> list[str]:
    """Find course codes in free text, e.g. ``CS 410`` or ``cs410``."""
    known = set(known_codes) if known_codes is not None else {c["course_id"] for c in get_courses()}
    codes: list[str] = []
    for match in _CODE_TOKEN_RE.finditer(text or ""):
        code = normalize_course_code(f"{match.group(1)} {match.group(2)}")
        uppercase = match.group(0).upper() == match.group(0)
        if not (uppercase or code in known):
            continue
        if code not in codes:
            codes.append(code)
    return codes


# ------------------------------------------------------------------------ PDF
def _detect_section(text: str) -> str:
    """Use the first heading-like line of a page as its section label."""
    for line in text.splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if not cleaned:
            continue
        if re.match(r"^page\s+\d+\s*[-\u2013\u2014?]", cleaned, re.IGNORECASE):
            return re.sub(r"^page\s+\d+\s*[-\u2013\u2014?]\s*", "", cleaned, flags=re.IGNORECASE)
        if cleaned.isupper() and len(cleaned) > 6:
            return cleaned.title()
        break
    return "General"


def load_pdf_pages(path: Path | None = None) -> list[dict[str, Any]]:
    """Extract text from each PDF page, keeping source and page metadata."""
    path = Path(path or get_settings().regulations_pdf)
    if not path.exists():
        raise FileNotFoundError(f"Academic regulations PDF not found: {path}")

    pages: list[dict[str, Any]] = []
    with pymupdf.open(str(path)) as document:
        for number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            pages.append(
                {
                    "page": number,
                    "text": text,
                    "section": _detect_section(text),
                    "source": path.name,
                }
            )
    return pages


# ------------------------------------------------------------- structured data
def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any) -> int | None:
    text = str(value).strip() if value is not None else ""
    return _to_int(text) if text else None


@lru_cache(maxsize=4)
def _read_courses(path: str) -> tuple[dict[str, Any], ...]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    courses: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        code = normalize_course_code(row.get("code") or row.get("course_id") or "")
        if not code:
            continue
        terms = [term.strip() for term in re.split(r"[;,]", str(row.get("offered_terms", ""))) if term.strip()]
        courses.append(
            {
                "course_id": code,
                "title": str(row.get("title", "")).strip(),
                "department": str(row.get("department", "")).strip(),
                "level": _to_int(row.get("level"), _to_int(code.split()[-1][:3])),
                "credits": _to_float(row.get("credits")),
                "description": str(row.get("description", "")).strip(),
                "prerequisites": str(row.get("prerequisites", "")).strip(),
                "corequisites": str(row.get("corequisites", "")).strip(),
                "offered_terms": terms,
                "schedule": str(row.get("schedule", "")).strip(),
                "instructor": str(row.get("instructor", "")).strip(),
                "capacity": _optional_int(row.get("capacity")),
                "seats_available": _optional_int(row.get("seats_available")),
            }
        )
    return tuple(courses)


@lru_cache(maxsize=4)
def _read_json_records(path: str) -> tuple[dict[str, Any], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(payload.get("records", payload.get("programs", payload.get("students", []))))


def load_courses(path: Path | None = None) -> list[dict[str, Any]]:
    path = Path(path or get_settings().courses_csv)
    return [dict(course) for course in _read_courses(str(path))]


def load_programs(path: Path | None = None) -> list[dict[str, Any]]:
    path = Path(path or get_settings().programs_json)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("programs", []))


def load_students(path: Path | None = None) -> list[dict[str, Any]]:
    path = Path(path or get_settings().students_json)
    if not Path(path).exists():
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    students = []
    for student in payload.get("students", []):
        record = dict(student)
        record["completed_courses"] = [normalize_course_code(code) for code in student.get("completed_courses", [])]
        students.append(record)
    return students


def get_courses() -> list[dict[str, Any]]:
    return load_courses()


def get_course(course_id: str) -> dict[str, Any] | None:
    wanted = normalize_course_code(course_id)
    for course in get_courses():
        if course["course_id"] == wanted:
            return course
    return None


def get_programs() -> list[dict[str, Any]]:
    return load_programs()


def get_program(name: str) -> dict[str, Any] | None:
    if not name:
        return None
    needle = name.strip().lower()
    programs = get_programs()
    for program in programs:
        if needle in {program["name"].lower(), str(program.get("id", "")).lower()}:
            return program
    for program in programs:
        if needle in program["name"].lower() or program["name"].lower() in needle:
            return program
    return None


def get_students() -> list[dict[str, Any]]:
    return load_students()


def course_document(course: dict[str, Any]) -> str:
    """Render a course as the text that gets embedded for semantic search."""
    return "\n".join(
        [
            f"{course['course_id']} - {course['title']}",
            f"Department: {course['department']}",
            f"Level: {course['level']}",
            f"Credits: {course['credits']:g}",
            f"Prerequisites: {course['prerequisites'] or 'None'}",
            f"Corequisites: {course['corequisites'] or 'None'}",
            f"Offered: {', '.join(course['offered_terms']) or 'Not specified'}",
            f"Schedule: {course['schedule'] or 'Not specified'}",
            f"Description: {course['description']}",
        ]
    )


# --------------------------------------------------------------------- indexing
def _digest(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_signature(settings: Settings | None = None) -> dict[str, Any]:
    """Data + chunking + embedder fingerprint used to detect a stale index."""
    settings = settings or get_settings()
    return {
        "courses_csv": _digest(settings.courses_csv),
        "regulations_pdf": _digest(settings.regulations_pdf),
        "chunker": f"{settings.chunk_size}/{settings.chunk_overlap}",
        "embedder": get_embedder(settings).identifier,
    }


def index_status(settings: Settings | None = None) -> dict[str, Any]:
    """Report whether the persisted index exists and matches current settings."""
    settings = settings or get_settings()
    manifest = read_manifest(settings)
    return {
        "exists": bool(manifest),
        "up_to_date": bool(manifest) and manifest.get("signature") == build_signature(settings),
        "counts": manifest.get("counts", {}),
        "built_at": manifest.get("built_at"),
        "embedder": manifest.get("signature", {}).get("embedder"),
    }


def build_index(force: bool = False, settings: Settings | None = None) -> dict[str, Any]:
    """Build the Chroma index, reusing it when nothing has changed."""
    settings = settings or get_settings()
    errors = settings.validation_errors()
    if errors:
        raise RuntimeError("; ".join(errors))

    signature = build_signature(settings)
    manifest = read_manifest(settings)
    if (
        not force
        and manifest.get("signature") == signature
        and manifest.get("counts", {}).get("documents", 0) > 0
    ):
        return {"status": "up_to_date", **manifest}

    embedder = get_embedder(settings)
    reset_collections(settings)

    # 1. Academic regulations PDF -> page-aware chunks -> embeddings.
    document_records: list[dict[str, Any]] = []
    for page in load_pdf_pages(settings.regulations_pdf):
        for chunk in split_page(
            page["text"],
            source=page["source"],
            page=page["page"],
            section=page["section"],
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ):
            document_records.append(
                {
                    "id": f"doc::{page['page']}::{chunk['metadata']['chunk_index']}",
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                }
            )
    add_records(
        DOCUMENTS_COLLECTION,
        document_records,
        embedder.embed_documents([record["text"] for record in document_records]),
        settings,
    )

    # 2. Course catalogue -> one searchable document per course.
    course_records = [
        {
            "id": f"course::{course['course_id']}",
            "text": course_document(course),
            "metadata": {
                "source": "courses.csv",
                "doc_type": "course",
                "course_code": course["course_id"],
                "department": course["department"],
                "level": int(course["level"]),
                "credits": float(course["credits"]),
            },
        }
        for course in get_courses()
    ]
    add_records(
        COURSES_COLLECTION,
        course_records,
        embedder.embed_documents([record["text"] for record in course_records]),
        settings,
    )

    payload = {
        "signature": signature,
        "counts": collection_counts(settings),
        "chunks": len(document_records),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    write_manifest(payload, settings)
    return {"status": "rebuilt", **payload}


def ensure_index(force: bool = False, settings: Settings | None = None) -> dict[str, Any]:
    """Build the index only when it is missing or stale (called on app startup)."""
    return build_index(force=force, settings=settings)
