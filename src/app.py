"""Streamlit user interface for the AI Academic Advisor.

The UI only collects input and renders results: every answer comes from the
LangGraph agent in ``agent.py``, and course search calls the real
``search_courses`` tool. Conversation history lives in Streamlit session state.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

if sys.version_info < (3, 11):
    raise SystemExit(
        "AI Academic Advisor requires Python 3.11 or newer; this interpreter is "
        f"{sys.version.split()[0]}.\n"
        "Recreate the environment with: py -3.11 -m venv .venv"
    )

# The modules live in src/; make the sibling imports work under any launcher
# (`streamlit run src/app.py`, `python src/app.py`, `python -m streamlit ...`).
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from config import get_settings
from ingestion import (
    build_index,
    ensure_index,
    get_courses,
    get_programs,
    get_students,
    index_status,
)

from agent import ask
from tools import search_courses

st.set_page_config(page_title="AI Academic Advisor", page_icon="🎓", layout="wide")

settings = get_settings()
errors = settings.validation_errors()
if errors:
    st.error("Configuration problem:\n\n" + "\n".join(f"- {error}" for error in errors))
    st.stop()


@st.cache_resource(show_spinner="Preparing the academic index...")
def prepare_index() -> dict[str, Any]:
    return ensure_index()


index_result = prepare_index()

students = get_students()
programs = get_programs()
courses = get_courses()
course_codes = [course["course_id"] for course in courses]
departments = sorted({course["department"] for course in courses})
program_names = [program["name"] for program in programs]

default_profile: dict[str, Any] = (
    dict(students[0])
    if students
    else {
        "student_id": "LOCAL-001",
        "name": "Student",
        "program": program_names[0] if program_names else "",
        "current_semester": 1,
        "completed_courses": [],
        "interests": [],
    }
)
if "profile" not in st.session_state:
    st.session_state.profile = default_profile


def load_sample_profile() -> None:
    """Copy the selected sample student into the editable profile."""
    labels = [f"{student['name']} ({student['student_id']})" for student in students]
    selected = st.session_state["sample_student"]
    chosen = labels.index(selected) if selected in labels else 0
    st.session_state.profile = dict(students[chosen])
    for key in ("profile_program", "profile_semester", "profile_completed", "profile_interests"):
        st.session_state.pop(key, None)


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for position, source in enumerate(sources, start=1):
            st.markdown(f"**[S{position}]** {source.get('label') or source.get('source', 'source')}")
            snippet = " ".join(str(source.get("text", "")).split())
            if snippet:
                st.caption(snippet[:400] + ("..." if len(snippet) > 400 else ""))


# ------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🎓 Academic profile")
    profile = st.session_state.profile

    if students:
        st.selectbox(
            "Load a sample student",
            [f"{student['name']} ({student['student_id']})" for student in students],
            key="sample_student",
            on_change=load_sample_profile,
        )

    program_index = program_names.index(profile["program"]) if profile.get("program") in program_names else 0
    program = st.selectbox("Programme", program_names, index=program_index, key="profile_program")
    semester = st.number_input(
        "Current semester",
        min_value=1,
        max_value=12,
        value=int(profile.get("current_semester") or 1),
        step=1,
        key="profile_semester",
    )
    completed = st.multiselect(
        "Completed courses",
        course_codes,
        default=[code for code in profile.get("completed_courses", []) if code in course_codes],
        key="profile_completed",
    )
    interests_text = st.text_input(
        "Academic interests (comma separated)",
        value=", ".join(profile.get("interests", [])),
        key="profile_interests",
    )

    student = {
        "student_id": profile.get("student_id", "LOCAL-001"),
        "name": profile.get("name", "Student"),
        "program": program,
        "current_semester": int(semester),
        "completed_courses": list(completed),
        "interests": [item.strip() for item in interests_text.split(",") if item.strip()],
    }

    st.caption(
        "Northgate University courses, programmes and regulations in this repository are "
        "fictional sample data for demonstration, not official policy."
    )

    st.divider()
    st.subheader("System")
    for label, value in settings.describe().items():
        st.caption(f"**{label}:** {value}")
    if settings.langsmith_enabled:
        st.link_button("View LangSmith traces", "https://smith.langchain.com")
    counts = index_result.get("counts", {})
    st.caption(
        f"**Index:** {index_result.get('status')} - "
        f"{counts.get('documents', 0)} document chunks, {counts.get('courses', 0)} courses"
    )
    if st.button("Rebuild index", use_container_width=True):
        with st.spinner("Re-embedding the academic documents..."):
            rebuilt = build_index(force=True)
        st.cache_resource.clear()
        st.success(f"Index rebuilt: {rebuilt['counts']}")
        st.rerun()

# --------------------------------------------------------------------------- tabs
chat_tab, courses_tab = st.tabs(["Advisor chat", "Course catalogue"])

with chat_tab:
    st.caption(
        f"Answering as **{student['name']}** - {student['program']}, semester "
        f"{student['current_semester']}, {len(student['completed_courses'])} completed courses."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources") or [])

    prompt = st.chat_input("Ask about courses, prerequisites, schedules, degree requirements or policies")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the catalogue and regulations..."):
                result = ask(prompt, student, thread_id=st.session_state.thread_id)
            st.markdown(result["answer"])
            render_sources(result["sources"])
            used = ", ".join(result["tools_used"]) or "none"
            st.caption(f"Intent: {result['intent']} | Tools: {used} | Answer mode: {result['mode']}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )

with courses_tab:
    with st.form("course_search_form"):
        query = st.text_input("Search the catalogue", placeholder="machine learning, databases, calculus")
        col_a, col_b, col_c = st.columns(3)
        department = col_a.selectbox("Department", ["Any"] + departments)
        min_level = col_b.selectbox("Minimum level", [0, 100, 200, 300, 400], index=0)
        limit = col_c.slider("Results", min_value=1, max_value=10, value=5)
        submitted = st.form_submit_button("Search courses")

    if submitted:
        if not query.strip():
            st.warning("Enter a search term first.")
        else:
            payload = json.loads(
                search_courses.invoke(
                    {
                        "query": query,
                        "department": "" if department == "Any" else department,
                        "min_level": int(min_level),
                        "limit": int(limit),
                    }
                )
            )
            results = payload.get("results") or []
            if not results:
                st.info(payload.get("note") or "No courses matched that search.")
            for course in results:
                st.subheader(f"{course['course_id']} - {course['title']}")
                st.caption(
                    f"{course['department']} | {course['credits']} credits | "
                    f"offered {', '.join(course['offered_terms']) or 'not listed'} | "
                    f"{course['schedule'] or 'schedule not listed'}"
                )
                st.markdown(f"**Prerequisites:** {course['prerequisites']}")
                if course.get("corequisites") not in (None, "", "None"):
                    st.markdown(f"**Corequisites:** {course['corequisites']}")
                st.write(course["description"])
                st.divider()

    with st.expander("Index and data status"):
        st.json(index_status())
