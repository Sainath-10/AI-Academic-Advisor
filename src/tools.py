"""The five academic tools used by the ReAct agent.

Each tool is a real Python function decorated with LangChain's ``@tool`` so the
chat model can select and call it. Every tool reads the ingested dataset and
returns JSON containing the result, a short note and citation records. Returning
citations from the tool is what lets the graph attach verified sources.

Tools:
    search_courses(query)
    get_degree_requirements(program)
    check_prerequisites(course_id, completed_courses)
    get_course_schedule(course_id)
    search_academic_documents(query)
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from ingestion import (
    get_course,
    get_courses,
    get_program,
    get_programs,
    normalize_course_code,
)
from retriever import search_course_documents, search_documents

from langchain_core.tools import tool

CATALOGUE_SOURCE = {
    "source": "courses.csv",
    "label": "Course catalogue (courses.csv)",
    "kind": "catalog",
}
PROGRAMS_SOURCE = {
    "source": "programs.json",
    "label": "Degree requirements (programs.json)",
    "kind": "program",
}


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


# ------------------------------------------------------- prerequisite expression
_TOKEN_RE = re.compile(
    r"\(|\)|\bAND\b|\bOR\b|,|;|[A-Za-z]{2,6}[\s-]?\d{3,4}[A-Za-z]?",
    re.IGNORECASE,
)


class _ExpressionParser:
    """Tiny recursive-descent parser for ``AND`` / ``OR`` / parentheses."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.index = 0

    def _peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _take(self) -> str:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def parse(self):
        node = self._parse_or()
        if self.index != len(self.tokens):
            raise ValueError(f"unexpected token '{self.tokens[self.index]}'")
        return node

    def _parse_or(self):
        node = self._parse_and()
        while (self._peek() or "").upper() == "OR":
            self._take()
            node = ("or", node, self._parse_and())
        return node

    def _parse_and(self):
        node = self._parse_atom()
        while (self._peek() or "").upper() in {"AND", ",", ";"}:
            self._take()
            node = ("and", node, self._parse_atom())
        return node

    def _parse_atom(self):
        token = self._peek()
        if token is None:
            raise ValueError("unexpected end of expression")
        if token == "(":
            self._take()
            node = self._parse_or()
            if self._peek() != ")":
                raise ValueError("unbalanced parentheses")
            self._take()
            return node
        if token.upper() in {")", "AND", "OR"}:
            raise ValueError(f"unexpected token '{token}'")
        return ("course", normalize_course_code(self._take()))


def _evaluate(node, completed: set[str]) -> tuple[bool, list[str], list[str]]:
    """Return ``(satisfied, missing, satisfied_by)`` for a parsed expression."""
    kind = node[0]
    if kind == "course":
        code = node[1]
        return (True, [], [code]) if code in completed else (False, [code], [])
    if kind == "and":
        left = _evaluate(node[1], completed)
        right = _evaluate(node[2], completed)
        return (
            left[0] and right[0],
            sorted(set(left[1] + right[1])),
            left[2] + right[2],
        )
    if kind == "or":
        left = _evaluate(node[1], completed)
        right = _evaluate(node[2], completed)
        if left[0]:
            return True, [], left[2]
        if right[0]:
            return True, [], right[2]
        return False, sorted(set(left[1] + right[1])), []
    raise ValueError(f"unknown node {kind}")


def evaluate_prerequisites(expression: str, completed_courses: Iterable[str]) -> dict[str, Any]:
    """Evaluate a prerequisite expression against completed courses."""
    completed = {
        normalize_course_code(code)
        for code in completed_courses
        if code and str(code).strip()
    }
    expression = (expression or "").strip()

    if not expression or expression.upper() in {"NONE", "N/A", "NA"}:
        return {
            "satisfied": True,
            "missing": [],
            "satisfied_by": [],
            "explanation": "No prerequisites are listed for this course.",
        }

    try:
        tree = _ExpressionParser(_TOKEN_RE.findall(expression)).parse()
        satisfied, missing, satisfied_by = _evaluate(tree, completed)
    except ValueError as error:
        return {
            "satisfied": False,
            "missing": [],
            "satisfied_by": [],
            "explanation": f"Could not interpret the prerequisite expression ({error}).",
        }

    return {
        "satisfied": satisfied,
        "missing": missing,
        "satisfied_by": sorted(set(satisfied_by)),
        "explanation": (
            "All prerequisite requirements are satisfied."
            if satisfied
            else "Missing prerequisite requirement(s): " + ", ".join(missing)
        ),
    }


# ------------------------------------------------------------------ tool helpers
def _course_summary(course: dict[str, Any]) -> dict[str, Any]:
    return {
        "course_id": course["course_id"],
        "title": course["title"],
        "department": course["department"],
        "level": course["level"],
        "credits": course["credits"],
        "prerequisites": course["prerequisites"] or "None",
        "corequisites": course["corequisites"] or "None",
        "offered_terms": course["offered_terms"],
        "schedule": course["schedule"],
        "instructor": course["instructor"],
        "seats_available": course["seats_available"],
        "description": course["description"],
    }


def _keyword_matches(query: str, limit: int) -> list[dict[str, Any]]:
    """Lexical catalogue search, used to complement vector search."""
    terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2]
    if not terms:
        return []
    scored = []
    for course in get_courses():
        haystack = " ".join(
            [
                course["course_id"],
                course["title"],
                course["department"],
                course["description"],
                course["prerequisites"],
            ]
        ).lower()
        score = sum(haystack.count(term) for term in terms)
        if score:
            scored.append((score, course))
    scored.sort(key=lambda item: (-item[0], item[1]["course_id"]))
    return [course for _score, course in scored[:limit]]


# ------------------------------------------------------------------------ tools
@tool
def search_courses(query: str, department: str = "", min_level: int = 0, limit: int = 5) -> str:
    """Search the course catalogue using semantic similarity plus optional filters.

    Use this to discover courses by topic, department or level.

    Args:
        query: Natural language description, e.g. "machine learning" or "databases".
        department: Optional department name, e.g. "Computer Science".
        min_level: Optional minimum course level, e.g. 300.
        limit: Maximum number of courses to return (1-10).
    """
    limit = max(1, min(int(limit or 5), 10))
    courses = get_courses()

    resolved_department: str | None = None
    if department.strip():
        needle = department.strip().lower()
        exact = [c["department"] for c in courses if c["department"].lower() == needle]
        partial = [c["department"] for c in courses if needle in c["department"].lower()]
        resolved_department = (exact or partial or [None])[0]

    hits = search_course_documents(
        query,
        k=limit,
        department=resolved_department,
        min_level=int(min_level) if int(min_level or 0) > 0 else None,
    )

    by_code = {course["course_id"]: course for course in courses}
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        course = by_code.get(hit.course_code)
        if course and course["course_id"] not in seen:
            ordered.append(course)
            seen.add(course["course_id"])

    for course in _keyword_matches(query, limit):
        if len(ordered) >= limit:
            break
        if course["course_id"] in seen:
            continue
        if resolved_department and course["department"].lower() != resolved_department.lower():
            continue
        if min_level and course["level"] < int(min_level):
            continue
        ordered.append(course)
        seen.add(course["course_id"])

    results = [_course_summary(course) for course in ordered[:limit]]
    return _dump(
        {
            "tool": "search_courses",
            "found": bool(results),
            "query": query,
            "filters": {"department": resolved_department, "min_level": min_level or None},
            "results": results,
            "sources": [CATALOGUE_SOURCE] if results else [],
            "note": "" if results else "No courses in the catalogue matched that search.",
        }
    )


@tool
def get_degree_requirements(program: str, completed_courses: list[str] | None = None) -> str:
    """Return the degree requirements for a programme and the student's progress.

    Args:
        program: Programme name or id, e.g. "Computer Science" or "ds-bs".
        completed_courses: Completed course codes; omit to skip the progress comparison.
    """
    record = get_program(program)
    if not record:
        available = ", ".join(item["name"] for item in get_programs())
        return _dump(
            {
                "tool": "get_degree_requirements",
                "found": False,
                "program": program,
                "results": [],
                "sources": [],
                "note": f"No programme named '{program}' was found. Available programmes: {available}.",
            }
        )

    comparing = completed_courses is not None
    completed = {normalize_course_code(code) for code in (completed_courses or []) if code}
    required = list(record.get("required_courses", []))
    remaining = [code for code in required if code not in completed] if comparing else required

    required_details = []
    for code in required:
        course = get_course(code)
        required_details.append(
            {
                "course_id": code,
                "title": course["title"] if course else "Unknown course",
                "credits": course["credits"] if course else None,
                "completed": (code in completed) if comparing else None,
            }
        )

    return _dump(
        {
            "tool": "get_degree_requirements",
            "found": True,
            "program": record["name"],
            "degree": record.get("degree"),
            "total_credits": record.get("total_credits"),
            "required_courses": required_details,
            "remaining_required_courses": remaining,
            "completed_credits_listed": sum(
                detail["credits"] or 0 for detail in required_details if detail["completed"]
            ),
            "elective_requirements": record.get("elective_requirements", []),
            "notes": record.get("notes", []),
            "sources": [
                {
                    **PROGRAMS_SOURCE,
                    "text": f"{record['name']} ({record.get('degree', '')}) requirements",
                }
            ],
            "note": (record.get("notes") or [""])[0],
        }
    )


@tool
def check_prerequisites(course_id: str, completed_courses: list[str] | None = None) -> str:
    """Check whether a student satisfies the prerequisites for a course.

    Args:
        course_id: Course code such as "CS 410".
        completed_courses: Completed course codes. Pass the student's full list;
            omit only if unknown.
    """
    course = get_course(course_id)
    if not course:
        return _dump(
            {
                "tool": "check_prerequisites",
                "found": False,
                "course_id": normalize_course_code(course_id),
                "sources": [],
                "note": f"Course '{course_id}' is not in the catalogue.",
            }
        )

    completed = list(completed_courses or [])
    result = evaluate_prerequisites(course["prerequisites"], completed)
    corequisite = (
        evaluate_prerequisites(course["corequisites"], completed) if course["corequisites"] else None
    )
    notes = []
    if completed_courses is None:
        notes.append(
            "No completed-course list was supplied, so the check assumes no courses are completed."
        )

    return _dump(
        {
            "tool": "check_prerequisites",
            "found": True,
            "course_id": course["course_id"],
            "title": course["title"],
            "prerequisites": course["prerequisites"] or "None",
            "corequisites": course["corequisites"] or "None",
            "completed_courses_considered": completed,
            "satisfied": result["satisfied"],
            "missing": result["missing"],
            "satisfied_by": result["satisfied_by"],
            "corequisite_ready": corequisite["satisfied"] if corequisite else None,
            "corequisite_missing": corequisite["missing"] if corequisite else [],
            "explanation": result["explanation"],
            "sources": [
                {
                    **CATALOGUE_SOURCE,
                    "text": f"{course['course_id']} prerequisites: {course['prerequisites'] or 'None'}",
                }
            ],
            "note": " ".join(notes),
        }
    )


@tool
def get_course_schedule(course_id: str) -> str:
    """Return meeting times, offered terms, instructor and seats for a course.

    Args:
        course_id: Course code such as "CS 350".
    """
    course = get_course(course_id)
    if not course:
        return _dump(
            {
                "tool": "get_course_schedule",
                "found": False,
                "course_id": normalize_course_code(course_id),
                "near_matches": [item["course_id"] for item in _keyword_matches(course_id, 3)],
                "sources": [],
                "note": f"Course '{course_id}' is not in the catalogue.",
            }
        )

    schedule = course["schedule"]
    return _dump(
        {
            "tool": "get_course_schedule",
            "found": True,
            "course_id": course["course_id"],
            "title": course["title"],
            "offered_terms": course["offered_terms"],
            "schedule": schedule,
            "instructor": course["instructor"],
            "credits": course["credits"],
            "capacity": course["capacity"],
            "seats_available": course["seats_available"],
            "sources": [
                {
                    **CATALOGUE_SOURCE,
                    "text": (
                        f"{course['course_id']} schedule: {schedule or 'not listed'}, "
                        f"offered {', '.join(course['offered_terms']) or 'not listed'}"
                    ),
                }
            ],
            "note": "" if schedule else "The catalogue does not list meeting times for this course.",
        }
    )


@tool
def search_academic_documents(query: str, limit: int = 4) -> str:
    """Semantic search of the university academic regulations PDF.

    Use for policy questions: grading, credit limits, probation, withdrawal,
    transfer credit, residency, graduation rules, advising, integrity.

    Args:
        query: Natural language question or keywords.
        limit: Number of regulation passages to retrieve (1-6).
    """
    limit = max(1, min(int(limit or 4), 6))
    chunks = search_documents(query, k=limit, doc_type="regulation")
    results = []
    sources = []
    for chunk in chunks:
        results.append(
            {
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "section": chunk.section,
                "score": round(chunk.score, 4),
            }
        )
        sources.append({**chunk.to_source()})

    return _dump(
        {
            "tool": "search_academic_documents",
            "found": bool(results),
            "query": query,
            "results": results,
            "sources": sources,
            "note": "" if results else "No matching passages were found in the academic regulations.",
        }
    )


ACADEMIC_TOOLS = [
    search_courses,
    get_degree_requirements,
    check_prerequisites,
    get_course_schedule,
    search_academic_documents,
]

TOOL_BY_NAME = {academic_tool.name: academic_tool for academic_tool in ACADEMIC_TOOLS}


def tool_sources(payload: Any) -> list[dict[str, Any]]:
    """Extract citation records from a tool's JSON payload."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, dict):
        return []
    return [source for source in payload.get("sources") or [] if isinstance(source, dict)]
