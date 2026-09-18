"""
Agent: ChatOllama + tool calling, ReAct style.

Design decision worth defending: the study plan shown on upload is produced by
running the study-plan tool DIRECTLY and then asking the model to phrase the
result. The tool genuinely runs the moment ingestion finishes - it does not
depend on a small local model choosing to call it. The numbers are therefore
always correct; the model only ever rewords them.

Follow-up questions go through the agent proper, which picks between searching
the transcript, re-running the plan, and fetching live listings.
"""
from __future__ import annotations

import inspect
import os

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from src_edu.tools import TOOLS, generate_study_plan

# create_agent moved between packages across versions; support both.
try:
    from langchain.agents import create_agent           # LangChain 1.x
except ImportError:                                      # pragma: no cover
    from langgraph.prebuilt import create_react_agent as create_agent

# qwen3:4b rather than 0.6b. The smaller model was measurably unreliable at tool
# calling. Override with OLLAMA_MODEL on a machine short of memory.
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

SYSTEM = """You are a study adviser working with ONE transcript the student just uploaded.

HARD RULES
1. You never calculate. The tools calculate. If you need a number - a score, an
   average, a count, study hours - call a tool. Never estimate one.
2. Never predict a future grade, a rank, or a probability of passing. You have
   no basis for it and the tools do not produce it.
3. Always say where something came from: "from your transcript" or "from the
   live listings feed".
4. If a tool reports it found nothing, say so plainly. Do not fill the gap from
   general knowledge about university courses.
5. Scores are out of 100. Study hours are per week.
6. Be encouraging but honest. A weak subject is weak; do not soften it into
   meaninglessness, and do not lecture the student about it either.
7. Be brief - three to six sentences unless a breakdown is asked for.
"""


def get_llm() -> ChatOllama:
    return ChatOllama(model=MODEL, temperature=0.1)


def make_retrieval_tool(retriever):
    """Wrap the retriever so the agent can search the transcript itself."""

    @tool
    def search_transcript(query: str) -> str:
        """Search the student's uploaded transcript for a specific subject, grade,
        semester or section. Use this for questions about what a particular
        subject scored, what was taken in a given semester, or anything quoted
        from the transcript itself. Returns the matching transcript extracts."""
        try:
            docs = retriever.invoke(query)
        except Exception as e:
            return "The transcript search failed: " + str(e)[:150]
        if not docs:
            return "Nothing in the transcript matched that query."
        return "\n\n".join(
            f"[{d.metadata.get('kind', 'section')}] {d.page_content}" for d in docs
        )

    return search_transcript


_PROMPT_KW = None


def prompt_kwarg() -> str:
    """Find what this installed version calls the system-prompt parameter."""
    global _PROMPT_KW
    if _PROMPT_KW is None:
        params = inspect.signature(create_agent).parameters
        _PROMPT_KW = next(
            (k for k in ("system_prompt", "prompt", "state_modifier", "messages_modifier")
             if k in params),
            "",
        )
    return _PROMPT_KW


def build_agent(retriever):
    tools = TOOLS + [make_retrieval_tool(retriever)]
    kw = prompt_kwarg()
    if kw:
        return create_agent(get_llm(), tools, **{kw: SYSTEM})
    return create_agent(get_llm(), tools)


def ask(agent, question: str, history: list | None = None) -> str:
    """Ask the agent one question. Never raises into the UI."""
    messages = list(history or []) + [HumanMessage(content=question)]
    try:
        result = agent.invoke({"messages": messages})
    except Exception as e:
        return ("The adviser could not be reached just now. Details: "
                + str(e)[:200])

    msgs = result.get("messages", []) if isinstance(result, dict) else []
    for m in reversed(msgs):
        text = getattr(m, "content", "")
        if isinstance(text, list):
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        if text and getattr(m, "type", "") in ("ai", "AIMessageChunk"):
            return str(text).strip()
    return "No answer came back. Try rephrasing the question."


def opening_summary(weak_threshold: float = 60.0) -> tuple[str, str]:
    """Run the study-plan tool directly, then have the model reword the result.

    Returns (raw_tool_output, plain_language_summary). The raw output is shown
    in the UI so the summary can be checked against its own working.
    """
    raw = generate_study_plan.invoke({"weak_threshold_pct": weak_threshold})

    try:
        llm = get_llm()
        out = llm.invoke([
            SystemMessage(content=(
                "Rewrite the study plan below for the student in three or four sentences. "
                "Use ONLY the numbers given - do not calculate anything new, do not add a "
                "figure that is not present, and do not predict future grades. Name the most "
                "urgent subject and say what the overall picture is. Be direct and encouraging."
            )),
            HumanMessage(content=raw),
        ])
        text = out.content if isinstance(out.content, str) else str(out.content)
        return raw, text.strip() or "See the plan below."
    except Exception:
        return raw, "The plain-language summary could not be generated, but the plan below is exact."
