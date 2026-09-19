"""
Tools for Track 3.

  1. generate_study_plan  - runs automatically the moment the transcript is
                            ingested. Deterministic Python, no LLM.
  2. search_internships   - live listings, with keywords taken FROM THE
                            TRANSCRIPT rather than typed by the student.
  3. search_class_chat    - retrieval over the class WhatsApp group, when one
                            has been uploaded. Optional; the tool is only bound
                            to the agent if a chat is present.

None of these ask the model to compute or decide anything. The model's only job
is to explain what they return.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_core.tools import tool

from src_edu.ingest import Course, WEAK_THRESHOLD, CRITICAL_THRESHOLD

# ---------------------------------------------------------------------------
# Session state. The agent should not be asked to carry a whole transcript
# through a tool argument, so the app sets it once after ingestion.
# ---------------------------------------------------------------------------
_COURSES: list[Course] = []


def set_session(courses: list[Course]) -> None:
    global _COURSES
    _COURSES = courses


def has_session() -> bool:
    return bool(_COURSES)


# ---------------------------------------------------------------------------
# Course -> skill keywords. Used both to suggest revision resources and to
# drive the internship search, so the search reflects what the student has
# actually studied rather than a keyword they typed.
# ---------------------------------------------------------------------------
SKILL_MAP: dict[str, list[str]] = {
    "data structures": ["algorithms", "problem solving"],
    "algorithms": ["algorithms", "problem solving"],
    "database": ["sql", "database"],
    "operating system": ["systems programming", "linux"],
    "computer network": ["networking"],
    "discrete mathematics": ["algorithms"],
    "software engineering": ["software engineer", "agile"],
    "theory of computation": ["computer science"],
    "web technolog": ["web developer", "javascript"],
    "machine learning": ["machine learning", "data science"],
    "artificial intelligence": ["artificial intelligence", "machine learning"],
    "compiler": ["systems programming"],
    "cloud": ["cloud", "devops"],
    "information security": ["security", "cybersecurity"],
    "mobile application": ["mobile developer", "android"],
    "python": ["python"],
    "java": ["java"],
}


def skills_for(subject: str) -> list[str]:
    s = subject.lower()
    for key, skills in SKILL_MAP.items():
        if key in s:
            return skills
    return []


@dataclass
class PlanItem:
    code: str
    subject: str
    semester: int
    score: float
    grade: str
    credits: float
    priority: int          # 1 is most urgent
    severity: str          # "critical" | "weak"
    hours_per_week: int
    reason: str


@dataclass
class StudyPlan:
    items: list[PlanItem]
    strong: list[str]
    average_score: float
    weak_count: int
    critical_count: int
    total_hours: int
    generated_at: str


def compute_plan(weak_threshold: float = WEAK_THRESHOLD) -> StudyPlan:
    """All of the study-plan arithmetic. No model involved anywhere in here."""
    weak = [c for c in _COURSES if c.total < weak_threshold]

    # Priority: worst score first; credits break ties, because a weak 4-credit
    # subject costs more than a weak 2-credit one.
    weak.sort(key=lambda c: (c.total, -c.credits))

    items: list[PlanItem] = []
    for i, c in enumerate(weak, start=1):
        critical = c.total < CRITICAL_THRESHOLD
        # Hours scale with the gap to the threshold and with credit weight.
        gap = max(0.0, weak_threshold - c.total)
        hours = int(min(10, max(2, round((gap / 10) + c.credits / 2))))
        items.append(PlanItem(
            code=c.code, subject=c.subject, semester=c.semester,
            score=c.total, grade=c.grade, credits=c.credits,
            priority=i,
            severity="critical" if critical else "weak",
            hours_per_week=hours,
            reason=(
                f"scored {c.total:g}/100 ({c.grade}), "
                + ("below the pass-risk line" if critical else "below the target line")
                + f" of {weak_threshold:g}, and carries {c.credits:g} credits"
            ),
        ))

    scored = [c.total for c in _COURSES if c.total > 0]
    strong = [c.subject for c in sorted(_COURSES, key=lambda c: -c.total)[:3] if c.total >= 70]

    return StudyPlan(
        items=items,
        strong=strong,
        average_score=round(sum(scored) / len(scored), 2) if scored else 0.0,
        weak_count=len(weak),
        critical_count=sum(1 for c in weak if c.total < CRITICAL_THRESHOLD),
        total_hours=sum(i.hours_per_week for i in items),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


def render_plan(p: StudyPlan) -> str:
    """The exact text the model is given. It rewords this; it never recomputes."""
    out = [
        f"STUDY PLAN (generated {p.generated_at}, from the uploaded transcript)",
        f"Subjects on the transcript: {len(_COURSES)}",
        f"Average score: {p.average_score:g} out of 100",
        f"Weak subjects: {p.weak_count} (of which {p.critical_count} are at risk of failing)",
        f"Suggested study load: {p.total_hours} hours per week in total",
        "",
    ]
    if not p.items:
        out.append("No subject falls below the threshold. Nothing is flagged for revision.")
    else:
        out.append("PRIORITISED WEAK AREAS:")
        for i in p.items:
            out.append(
                f"{i.priority}. {i.subject} ({i.code}, semester {i.semester}) - "
                f"{i.severity.upper()} - {i.hours_per_week} hrs/week. "
                f"Reason: {i.reason}."
            )
    if p.strong:
        out.append("")
        out.append("STRONGEST SUBJECTS: " + ", ".join(p.strong))
    out.append("")
    out.append("These figures come from the transcript only. No prediction is made.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Tool 1 (required): study plan - runs automatically on ingestion
# ---------------------------------------------------------------------------
@tool
def generate_study_plan(weak_threshold_pct: float = 60.0) -> str:
    """Build a prioritised study plan from the student's uploaded transcript.

    Uses deterministic arithmetic over the transcript that was just ingested.
    For every subject scoring below weak_threshold_pct (a PERCENTAGE out of 100,
    default 60) it returns the subject, its score, its grade, how urgent it is,
    a suggested number of study hours per week, and the reason it was flagged.
    Also returns the overall average and the student's strongest subjects.

    Scores are out of 100. Hours are per week. Call this for any question about
    what to revise, which subjects are weak, or how the student is performing
    overall. It makes no prediction and gives no grade forecast.
    """
    if not has_session():
        return "No transcript has been uploaded yet, so there is nothing to plan."
    return render_plan(compute_plan(weak_threshold_pct))


# ---------------------------------------------------------------------------
# Tool 2 (required): live internship search, keyed off the transcript
# ---------------------------------------------------------------------------
JOBS_ENDPOINT = "https://remotive.com/api/remote-jobs"
JOBS_TIMEOUT = 12


def transcript_keywords(limit: int = 4) -> list[str]:
    """Skill keywords drawn from the student's STRONGEST subjects.

    The brief is explicit that the search must use keywords extracted from the
    transcript rather than typed by the student. Strongest-first is the
    deliberate choice: a student is more employable in what they did well, and
    the weak subjects are already covered by the study plan.
    """
    ordered = sorted(_COURSES, key=lambda c: -c.total)
    seen: list[str] = []
    for c in ordered:
        if c.total < WEAK_THRESHOLD:
            continue
        for skill in skills_for(c.subject):
            if skill not in seen:
                seen.append(skill)
            if len(seen) >= limit:
                return seen
    return seen or ["computer science"]


def _fetch_jobs(keyword: str, limit: int) -> list[dict]:
    url = JOBS_ENDPOINT + "?" + urllib.parse.urlencode({"search": keyword, "limit": limit})
    req = urllib.request.Request(url, headers={"User-Agent": "sou-study-agent/1.0"})
    with urllib.request.urlopen(req, timeout=JOBS_TIMEOUT) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    return payload.get("jobs", []) or []


@tool
def search_internships(max_results: int = 6) -> str:
    """Search live internship and graduate listings matched to this transcript.

    Takes NO search term from the user. The keywords are extracted from the
    student's own transcript - specifically their strongest subjects - so the
    results reflect what they have actually studied rather than whatever they
    happened to type. Returns live listings with role, company, location and a
    link, plus the keywords that were used, so the student can see why each
    result appeared.

    Call this when the student asks about jobs, internships, placements or what
    they could apply for. Returns at most max_results listings.
    """
    if not has_session():
        return "No transcript has been uploaded yet, so there is nothing to match against."

    keywords = transcript_keywords()
    lines = ["LIVE LISTINGS (searched using keywords taken from the transcript: "
             + ", ".join(keywords) + ")", ""]

    found = 0
    errors: list[str] = []
    for kw in keywords:
        if found >= max_results:
            break
        try:
            jobs = _fetch_jobs(kw, max_results)
        except Exception as e:                      # network, timeout, bad JSON
            errors.append(f"{kw}: {str(e)[:80]}")
            continue
        for j in jobs:
            if found >= max_results:
                break
            lines.append(
                f"- {j.get('title', 'Untitled')} at {j.get('company_name', 'unknown company')} "
                f"({j.get('candidate_required_location') or 'location not stated'}) "
                f"[matched on: {kw}] {j.get('url', '')}"
            )
            found += 1

    if not found:
        lines.append("No live listings came back for these keywords.")
        if errors:
            lines.append("The job feed could not be reached: " + "; ".join(errors))
        lines.append("This is a live feed problem, not a statement about the student's prospects.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3 (optional): the class WhatsApp group
#
# Only bound to the agent when a chat has actually been uploaded. That is the
# same principle the portal uses for its finance tools: a capability the caller
# does not have is not described to the model at all, so no amount of clever
# phrasing can reach it. Here it also means the model cannot promise a student
# an answer from a group chat that was never provided.
# ---------------------------------------------------------------------------
_CHAT_RETRIEVER = None
_CHAT_SUMMARY: str = ""


def set_class_chat(retriever, summary: str = "") -> None:
    global _CHAT_RETRIEVER, _CHAT_SUMMARY
    _CHAT_RETRIEVER = retriever
    _CHAT_SUMMARY = summary


def clear_class_chat() -> None:
    """Forget the uploaded chat. The app calls this when the student removes it."""
    set_class_chat(None, "")


def has_class_chat() -> bool:
    return _CHAT_RETRIEVER is not None


@tool
def search_class_chat(query: str) -> str:
    """Search the class WhatsApp group the student uploaded.

    Use this for anything that would have been announced in the class group
    rather than printed in a document: exam and viva dates, submission
    deadlines, room or venue changes, what to bring tomorrow, cancelled
    lectures, or "what did sir say about X". Pass the student's question, or
    the key phrase from it, as the query.

    Returns the matching messages with the sender and the date, so the student
    can see exactly which message an answer came from. Phone numbers have been
    removed from this data; senders who appeared only as numbers are shown as
    "Member 1", "Member 2" and so on.

    If the messages returned do not actually answer the question, say so. Do
    not fill the gap from general knowledge - a wrong exam date is worse than
    no exam date.
    """
    if not has_class_chat():
        return "No class group chat has been uploaded, so there is nothing to search."

    try:
        docs = _CHAT_RETRIEVER.invoke(query)
    except Exception as e:
        return f"The class chat could not be searched: {str(e)[:160]}"

    if not docs:
        return "No message in the uploaded class group matched that."

    lines = ["MESSAGES FROM THE CLASS GROUP (most relevant first):", ""]
    for d in docs:
        kind = d.metadata.get("kind", "message")
        date = d.metadata.get("date", "unknown date")
        lines.append(f"[{kind}, {date}]")
        lines.append(d.page_content.strip())
        lines.append("")
    lines.append(
        "These are the student's own uploaded messages. If none of them states "
        "the answer, say that it was not found in the group rather than guessing."
    )
    return "\n".join(lines)


def tools_for_session() -> list:
    """The toolset for this session.

    search_class_chat is added only when a chat has been uploaded, so the model
    is never told about a source that does not exist.
    """
    tools = [generate_study_plan, search_internships]
    if has_class_chat():
        tools.append(search_class_chat)
    return tools


TOOLS = [generate_study_plan, search_internships]
