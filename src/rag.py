"""Retrieval-Augmented Generation.

The RAG pipeline:

1. take the student question and the chunks/tool results already collected by the graph,
2. build one grounded prompt (system rules + student profile + retrieved context),
3. ask the configured chat model,
4. append the verified source list.

Without an API key no model is called: ``compose_extractive_answer`` builds the
answer directly from the tool observations and labels itself as offline mode.
Nothing here invents university facts; the prompt forbids it and the extractive
path only quotes retrieved text.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from config import get_settings

from langchain_core.messages import HumanMessage

logger = logging.getLogger("advisor.rag")

SYSTEM_PROMPT = """You are the AI Academic Advisor for Northgate University.

Answer student questions about courses, prerequisites, degree requirements,
schedules and academic regulations.

Grounding rules:
1. Base every factual claim on the RETRIEVED CONTEXT or TOOL RESULTS in this prompt.
   Do not use general knowledge about other universities.
2. Cite the supporting source inline with its label, for example [S1] or [S3].
3. If the material does not contain the answer, say clearly: "I couldn't find that
   information in the available university documents." Never invent policies,
   deadlines, credit limits or course requirements.
4. Do not add a "Sources" list; the application appends the verified list.
5. When asked about eligibility, check the student's completed courses explicitly
   and name the missing requirement.
6. Use the student's programme, semester and interests when recommending courses.
7. Be concise, structured and use Markdown. Never claim an action you did not perform.
"""

OFFLINE_NOTE = (
    "This answer was assembled directly from the retrieved documents and catalogue "
    "records by the deterministic offline planner (no LLM is configured)."
)


def create_chat_model(tools: list[Any] | None = None, temperature: float = 0.1):
    """Create the configured chat model, optionally bound to tools.

    Returns ``None`` when no API key is configured; callers then use the offline path.
    """
    settings = get_settings()
    if not settings.llm_available:
        return None

    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.llm_api_key,
            temperature=temperature,
            timeout=60,
            max_retries=1,
        )
    elif settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        model = ChatGroq(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=temperature,
            timeout=60,
            max_retries=1,
        )
    else:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or None,
            temperature=temperature,
            timeout=60,
            max_retries=1,
        )
    return model.bind_tools(tools) if tools else model


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def student_block(student: dict[str, Any] | None) -> str:
    if not student:
        return "No student profile is available."
    completed = ", ".join(student.get("completed_courses") or []) or "none"
    interests = ", ".join(student.get("interests") or []) or "not specified"
    return "\n".join(
        [
            f"Student ID: {student.get('student_id', 'unknown')}",
            f"Name: {student.get('name', 'unknown')}",
            f"Programme: {student.get('program', 'undeclared')}",
            f"Current semester: {student.get('current_semester', 'unknown')}",
            f"Completed courses: {completed}",
            f"Academic interests: {interests}",
        ]
    )


def context_block(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "RETRIEVED CONTEXT: (no document chunks were retrieved)"
    lines = ["RETRIEVED CONTEXT:"]
    for index, source in enumerate(sources, start=1):
        label = source.get("label") or source.get("source", "source")
        lines.append(f"[S{index}] {label}")
        lines.append(str(source.get("text", ""))[:1400])
        lines.append("")
    return "\n".join(lines).strip()


def tool_block(payloads: list[dict[str, Any]]) -> str:
    if not payloads:
        return ""
    blocks = ["TOOL RESULTS:"]
    for payload in payloads:
        blocks.append(f"{payload.get('tool', 'tool')}:\n{json.dumps(payload, ensure_ascii=False, default=str)[:4000]}")
    return "\n\n".join(blocks)


def build_prompt(
    question: str,
    student: dict[str, Any] | None,
    sources: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
) -> str:
    """Assemble the grounded prompt sent to the chat model."""
    parts = [
        SYSTEM_PROMPT,
        "STUDENT PROFILE\n" + student_block(student),
        context_block(sources),
    ]
    tools_text = tool_block(payloads)
    if tools_text:
        parts.append(tools_text)
    parts.append("STUDENT QUESTION\n" + question)
    parts.append("Write the final answer for the student, citing sources inline with [S#] labels.")
    return "\n\n".join(parts)


def append_sources(answer: str, sources: list[dict[str, Any]]) -> str:
    """Append the verified source list collected by the graph."""
    if not sources:
        return answer.strip()
    lines = ["", "---", "**Sources**"]
    for index, source in enumerate(sources, start=1):
        lines.append(f"- [S{index}] {source.get('label') or source.get('source', 'source')}")
    return answer.strip() + "\n" + "\n".join(lines)


# ------------------------------------------------------------ offline rendering
def _course_bullets(payload: dict[str, Any]) -> str:
    results = payload.get("results") or []
    if not results:
        return ""
    lines = []
    for course in results[:6]:
        lines.append(
            f"- **{course.get('course_id')} {course.get('title')}** "
            f"({course.get('credits')} credits, {course.get('schedule') or 'schedule not listed'}) "
            f"- prerequisites: {course.get('prerequisites') or 'None'}."
        )
    return "\n".join(lines)


def _prerequisite_summary(payload: dict[str, Any]) -> str:
    if not payload.get("found"):
        return payload.get("note", "That course was not found in the catalogue.")
    status = (
        "**eligible** - all prerequisites are satisfied."
        if payload.get("satisfied")
        else "**not yet eligible** - missing: " + (", ".join(payload.get("missing") or []) or "an unspecified requirement") + "."
    )
    lines = [
        f"**{payload.get('course_id')} {payload.get('title')}**",
        f"- Prerequisite requirement: `{payload.get('prerequisites')}`",
        f"- Your status: {status}",
        f"- {payload.get('explanation', '')}",
    ]
    if payload.get("corequisites") and payload.get("corequisites") != "None":
        state = "met" if payload.get("corequisite_ready") else "not met"
        lines.append(f"- Corequisite `{payload.get('corequisites')}`: {state}.")
    return "\n".join(lines)


def _degree_summary(payload: dict[str, Any]) -> str:
    if not payload.get("found"):
        return payload.get("note", "That programme was not found.")
    required = payload.get("required_courses") or []
    remaining = payload.get("remaining_required_courses") or []
    lines = [
        f"**{payload.get('program')} - {payload.get('degree')}**",
        f"Total credits required: {payload.get('total_credits')}.",
        f"Required courses completed: {len(required) - len(remaining)} of {len(required)}.",
    ]
    if remaining:
        lines.append("Outstanding required courses: " + ", ".join(remaining) + ".")
    else:
        lines.append("All listed required courses are complete.")
    for elective in payload.get("elective_requirements") or []:
        lines.append(
            f"- Elective requirement: {elective.get('name')} "
            f"({elective.get('courses_required')} courses, level {elective.get('min_level', 'any')}+)."
        )
    if payload.get("notes"):
        lines.append("Programme notes: " + " ".join(payload["notes"]))
    return "\n".join(lines)


def _schedule_summary(payload: dict[str, Any]) -> str:
    if not payload.get("found"):
        return payload.get("note", "That course was not found.")
    lines = [
        f"**{payload.get('course_id')} {payload.get('title')}**",
        f"- Meeting time: {payload.get('schedule') or 'not listed'}",
        f"- Offered: {', '.join(payload.get('offered_terms') or []) or 'not listed'}",
        f"- Instructor: {payload.get('instructor') or 'not listed'}",
    ]
    if payload.get("seats_available") is not None:
        lines.append(f"- Seats available: {payload.get('seats_available')} of {payload.get('capacity')}")
    return "\n".join(lines)


def _documents_summary(payload: dict[str, Any]) -> str:
    results = payload.get("results") or []
    if not results:
        return ""
    lines = ["Based on the academic regulations:"]
    for entry in results[:3]:
        text = " ".join(str(entry.get("text", "")).split())
        if len(text) > 700:
            text = text[:700].rstrip() + "..."
        page = f"p. {entry.get('page')}" if entry.get("page") is not None else "page unknown"
        lines.append(f"> {text}\n\n({entry.get('source')}, {page})")
    return "\n\n".join(lines)


def compose_extractive_answer(payloads: list[dict[str, Any]]) -> str:
    """Build a grounded answer from tool observations only (no LLM required)."""
    handlers = {
        "check_prerequisites": _prerequisite_summary,
        "get_degree_requirements": _degree_summary,
        "get_course_schedule": _schedule_summary,
        "search_courses": _course_bullets,
        "search_academic_documents": _documents_summary,
    }

    sections = []
    for payload in payloads:
        handler = handlers.get(payload.get("tool", ""))
        if handler:
            rendered = handler(payload)
            if rendered:
                sections.append(rendered)

    if not sections:
        return (
            "I couldn't find that information in the available university documents or "
            "course catalogue. I can help with course search, prerequisites, schedules, "
            "degree requirements and academic regulations."
        )
    return "\n\n".join(sections) + "\n\n_" + OFFLINE_NOTE + "_"


def generate_answer(
    question: str,
    student: dict[str, Any] | None,
    sources: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
) -> tuple[str, str]:
    """Return ``(answer, mode)`` where mode is ``"llm"`` or ``"offline"``."""
    model = create_chat_model()
    if model is not None:
        try:
            response = model.invoke([HumanMessage(content=build_prompt(question, student, sources, payloads))])
            answer = _message_text(response).strip()
            if answer:
                return answer, "llm"
        except Exception as error:  # network/auth errors must not break the app
            logger.warning("LLM answer generation failed, using extractive fallback: %s", error)

    return compose_extractive_answer(payloads), "offline"
