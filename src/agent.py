"""LangGraph workflow + ReAct agent.

    START -> understand_query -> (greeting)     -> direct_answer -> END
                              -> (everything else) -> retrieve_context -> agent
                                                                          |
                                                          tool calls? <---+
                                                                |
                                                                v
                                                          execute_tools
                                                                |
                                                                +--> agent (loop, max 3 rounds)
                                                          no tool calls
                                                                |
                                                                v
                                                        generate_answer -> END

The ``agent`` node is the ReAct reason step: with an LLM configured it binds the
five tools and decides which to call; the ``execute_tools`` node runs them and
returns observations. Without an API key a deterministic planner picks the same
tools from the classified intent, so the whole pipeline still runs offline.

Conversation state is stored in a LangGraph ``MemorySaver`` keyed by ``thread_id``,
which is how follow-up questions keep their context.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from functools import lru_cache
from typing import Annotated, Any, Literal, TypedDict

from config import get_settings
from ingestion import extract_course_codes, get_programs

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from rag import append_sources, create_chat_model, generate_answer
from retriever import search_documents
from tools import ACADEMIC_TOOLS, TOOL_BY_NAME, tool_sources

logger = logging.getLogger("advisor.agent")

Intent = Literal[
    "greeting",
    "course_search",
    "prerequisites",
    "course_eligibility",
    "degree_requirements",
    "schedule",
    "recommendation",
    "policy",
    "general",
]

MAX_TOOL_ROUNDS = 3


class AdvisorState(TypedDict, total=False):
    """State passed between graph nodes."""

    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    student: dict[str, Any]
    intent: str
    rewritten_query: str
    retrieved: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    tool_rounds: int
    tools_used: list[str]
    answer: str
    mode: str


# --------------------------------------------------------------------- utilities
def _text_of(message: AnyMessage) -> str:
    content = getattr(message, "content", "")
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


def _merge_sources(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(existing)
    seen = {(source.get("source"), source.get("page"), source.get("label")) for source in merged}
    for source in new:
        key = (source.get("source"), source.get("page"), source.get("label"))
        if key in seen:
            continue
        seen.add(key)
        record = dict(source)
        record.setdefault("kind", "document")
        record.setdefault("label", str(source.get("source", "source")))
        merged.append(record)
    return merged


def _last_ai_message(messages: list[AnyMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _parse_tool_payloads(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    payloads = []
    for message in messages:
        if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payload.setdefault("tool", getattr(message, "name", "tool"))
            payloads.append(payload)
    return payloads


# ------------------------------------------------------------ query understanding
def classify_question(question: str) -> dict[str, str]:
    """Rule-based intent classifier (the offline planner and the LLM fallback)."""
    text = (question or "").lower().strip()
    codes = extract_course_codes(question or "")

    if re.fullmatch(r"(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you)[!. ]*", text):
        return {"intent": "greeting", "rewritten_query": question}

    def has(*keywords: str) -> bool:
        return any(keyword in text for keyword in keywords)

    if has("prereq", "prerequisit", "eligible", "eligibility", "can i register", "can i take", "am i allowed"):
        intent = "prerequisites" if codes else "course_eligibility"
    elif has("degree requirement", "degree requirements", "requirements for", "graduate", "graduation",
             "my major", "remaining requirement"):
        intent = "degree_requirements"
    elif has("schedule", "what time", "when does", "when is", "meeting time", "meets", "offered", "timetable"):
        intent = "schedule"
    elif has("recommend", "should i take", "next semester", "plan my", "course plan", "what courses"):
        intent = "recommendation"
    elif has("gpa", "probation", "withdraw", "credit limit", "credit load", "transfer credit", "residency",
             "pass/fail", "incomplete", "repeat a course", "academic integrity", "regulation", "policy",
             "leave of absence", "part-time", "summer", "attendance"):
        intent = "policy"
    elif codes:
        intent = "course_search"
    else:
        intent = "general"

    return {"intent": intent, "rewritten_query": question}


def _understand_query(state: AdvisorState) -> dict[str, Any]:
    question = state.get("question", "")
    history = [message for message in state.get("messages", [])[:-1] if isinstance(message, HumanMessage)]
    previous = _text_of(history[-1]) if history else ""

    result = classify_question(question)
    # Resolve short follow-ups ("what about its prerequisites?") against the previous turn.
    if previous and len(question.split()) <= 8 and re.search(r"\b(it|that|those|this|them)\b", question.lower()):
        result["rewritten_query"] = f"{previous} {question}".strip()

    return {
        "intent": result["intent"],
        "rewritten_query": result["rewritten_query"],
        "mode": "offline",
    }


# ---------------------------------------------------------------------- retrieval
def _retrieve_context(state: AdvisorState) -> dict[str, Any]:
    settings = get_settings()
    query = state.get("rewritten_query") or state.get("question", "")
    chunks = search_documents(query, k=settings.retrieval_k, doc_type="regulation")
    sources = _merge_sources(state.get("sources", []), [chunk.to_source() for chunk in chunks])
    return {
        "retrieved": [
            {
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "section": chunk.section,
                "score": round(chunk.score, 4),
            }
            for chunk in chunks
        ],
        "sources": sources,
    }


# ----------------------------------------------------------------- ReAct planning
def _target_program(question: str, student: dict[str, Any]) -> str:
    """Pick the programme named in the question, else the student's programme."""
    lowered = (question or "").lower()
    for record in get_programs():
        if record["name"].lower() in lowered or str(record.get("id", "")).lower() in lowered:
            return record["name"]
    return (student or {}).get("program", "")


def plan_offline_tool_calls(state: AdvisorState) -> list[dict[str, Any]]:
    """Deterministic tool selection used when no LLM is configured.

    Mirrors one ReAct step: choose tools for the classified intent, then answer
    from the observations. Only real tools from ``ACADEMIC_TOOLS`` are selected.
    """
    if state.get("tool_rounds", 0) > 0:
        return []

    question = state.get("question", "")
    rewritten = state.get("rewritten_query") or question
    student = state.get("student") or {}
    intent = state.get("intent", "general")
    completed = list(student.get("completed_courses") or [])
    codes = extract_course_codes(rewritten)

    calls: list[dict[str, Any]] = []

    def add(name: str, args: dict[str, Any]) -> None:
        calls.append(
            {
                "name": name,
                "args": args,
                "id": f"offline_{len(calls) + 1}_{name}",
                "type": "tool_call",
            }
        )

    if intent in {"prerequisites", "course_eligibility"}:
        if codes:
            add("check_prerequisites", {"course_id": codes[0], "completed_courses": completed})
            add("search_academic_documents", {"query": "prerequisites and corequisites policy", "limit": 2})
        else:
            add("search_courses", {"query": rewritten, "limit": 5})
    elif intent == "schedule":
        if codes:
            add("get_course_schedule", {"course_id": codes[0]})
        else:
            add("search_courses", {"query": rewritten, "limit": 5})
    elif intent == "degree_requirements":
        add(
            "get_degree_requirements",
            {"program": _target_program(question, student) or "Computer Science", "completed_courses": completed},
        )
    elif intent == "recommendation":
        interests = ", ".join(student.get("interests") or [])
        add("search_courses", {"query": f"{rewritten} {interests}".strip(), "limit": 5})
        add(
            "get_degree_requirements",
            {"program": _target_program(question, student) or "Computer Science", "completed_courses": completed},
        )
    elif intent in {"policy", "general"}:
        add("search_academic_documents", {"query": rewritten, "limit": 4})
    else:
        add("search_courses", {"query": rewritten, "limit": 5})

    return calls


def _agent(state: AdvisorState) -> dict[str, Any]:
    """ReAct reason step: decide on tool calls, or conclude the reasoning."""
    model = create_chat_model(tools=ACADEMIC_TOOLS)
    if model is not None:
        student = state.get("student") or {}
        system = SystemMessage(
            content=(
                "You are the AI Academic Advisor for Northgate University. Choose tools when they "
                "are needed and answer only from tool results and retrieved documents.\n"
                f"Programme: {student.get('program', 'unknown')}\n"
                f"Completed courses: {', '.join(student.get('completed_courses') or []) or 'none'}\n"
                f"Interests: {', '.join(student.get('interests') or []) or 'not specified'}\n"
                f"Current semester: {student.get('current_semester', 'unknown')}\n"
                "Pass the student's completed-course list when calling check_prerequisites. "
                "When you have enough information, reply without tool calls."
            )
        )
        try:
            response = model.invoke([system, *state.get("messages", [])])
            return {"messages": [response], "mode": "llm"}
        except Exception as error:  # network/auth/model errors fall back to the offline planner
            logger.warning("LLM agent call failed, using offline planner: %s", error)

    tool_calls = plan_offline_tool_calls(state)
    return {
        "messages": [AIMessage(content="" if tool_calls else "Plan complete.", tool_calls=tool_calls)],
        "mode": "offline",
    }


# ------------------------------------------------------------------ tool execution
def _execute_tools(state: AdvisorState) -> dict[str, Any]:
    last = _last_ai_message(state.get("messages", []))
    tool_calls = list(getattr(last, "tool_calls", []) or []) if last else []

    messages: list[ToolMessage] = []
    sources = list(state.get("sources", []))
    used = list(state.get("tools_used", []))

    for call in tool_calls:
        name = call.get("name", "")
        args = call.get("args") or {}
        academic_tool = TOOL_BY_NAME.get(name)
        if academic_tool is None:
            content = json.dumps({"error": f"Unknown tool: {name}", "found": False, "sources": []})
        else:
            try:
                content = academic_tool.invoke(args)
            except Exception as error:  # a failing tool is an observation, not a crash
                logger.exception("Tool %s failed", name)
                content = json.dumps({"error": str(error), "found": False, "tool": name, "sources": []})
        if not isinstance(content, str):
            content = json.dumps(content, default=str)
        messages.append(
            ToolMessage(
                content=content,
                tool_call_id=call.get("id") or f"call_{len(used)}",
                name=name,
            )
        )
        sources = _merge_sources(sources, tool_sources(content))
        used.append(name)

    return {
        "messages": messages,
        "sources": sources,
        "tools_used": list(dict.fromkeys(used)),
        "tool_rounds": int(state.get("tool_rounds", 0)) + 1,
    }


def _route_after_agent(state: AdvisorState) -> str:
    last = _last_ai_message(state.get("messages", []))
    if last is not None and getattr(last, "tool_calls", None):
        if int(state.get("tool_rounds", 0)) < MAX_TOOL_ROUNDS:
            return "execute_tools"
    return "generate_answer"


# --------------------------------------------------------------------- answering
def _generate_answer(state: AdvisorState) -> dict[str, Any]:
    sources = list(state.get("sources", []))
    payloads = _parse_tool_payloads(state.get("messages", []))
    answer, mode = generate_answer(state.get("question", ""), state.get("student"), sources, payloads)
    final = append_sources(answer, sources)
    return {
        "answer": final,
        "sources": sources,
        "mode": mode,
        "messages": [AIMessage(content=final)],
    }


def _direct_answer(state: AdvisorState) -> dict[str, Any]:
    student = state.get("student") or {}
    name = student.get("name") or "there"
    program = student.get("program")
    message = (
        f"Hello {name}! I'm your AI Academic Advisor"
        + (f" for the {program} programme" if program else "")
        + ". Ask me about courses, prerequisites, schedules, degree requirements or academic "
        'regulations. For example: "What are the prerequisites for CS 410?" or '
        '"Do I still need any required courses?"'
    )
    return {"answer": message, "messages": [AIMessage(content=message)], "sources": []}


# ----------------------------------------------------------------------- routing
def _route_after_understanding(state: AdvisorState) -> str:
    return "direct_answer" if state.get("intent") == "greeting" else "retrieve_context"


def build_graph():
    """Compile the LangGraph workflow used by the application."""
    graph = StateGraph(AdvisorState)

    graph.add_node("understand_query", _understand_query)
    graph.add_node("retrieve_context", _retrieve_context)
    graph.add_node("agent", _agent)
    graph.add_node("execute_tools", _execute_tools)
    graph.add_node("generate_answer", _generate_answer)
    graph.add_node("direct_answer", _direct_answer)

    graph.add_edge(START, "understand_query")
    graph.add_conditional_edges(
        "understand_query",
        _route_after_understanding,
        {"direct_answer": "direct_answer", "retrieve_context": "retrieve_context"},
    )
    graph.add_edge("retrieve_context", "agent")
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"execute_tools": "execute_tools", "generate_answer": "generate_answer"},
    )
    graph.add_edge("execute_tools", "agent")
    graph.add_edge("generate_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile(checkpointer=MemorySaver())


@lru_cache(maxsize=1)
def get_advisor_graph():
    return build_graph()


def build_initial_state(question: str, student: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "student": student or {},
        "sources": [],
        "tool_rounds": 0,
        "tools_used": [],
    }


def ask(
    question: str,
    student: dict[str, Any] | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Run one turn through the LangGraph agent and return the grounded result."""
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
    final_state = get_advisor_graph().invoke(build_initial_state(question, student), config)
    return {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "intent": final_state.get("intent", ""),
        "tools_used": final_state.get("tools_used", []),
        "mode": final_state.get("mode", "offline"),
    }
