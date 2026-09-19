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
import re

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from src_edu.tools import (
    tools_for_session, generate_study_plan, has_class_chat, search_class_chat,
)

# create_agent moved between packages across versions; support both.
try:
    from langchain.agents import create_agent           # LangChain 1.x
except ImportError:                                      # pragma: no cover
    from langgraph.prebuilt import create_react_agent as create_agent

# qwen3:8b. The smaller sizes were measurably unreliable at tool calling - 1.7b
# would answer an exam-date question from its own weights rather than calling
# the retrieval tool, which is the one failure this app must not have. Override
# with OLLAMA_MODEL on a machine short of memory or disk; 4b is a workable
# fallback, and answer_from_chat runs retrieval in Python either way so the
# class-group path does not depend on the model choosing correctly.
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# ---------------------------------------------------------------------------
# Model provider.
#
# Ollama is the default and stays the default: the pipeline this project is
# marked against names ChatOllama, and the whole thing genuinely runs offline
# on a laptop, which is worth keeping.
#
# But a 1.7B model on a laptop is measurably unreliable at deciding to call a
# tool, and pulling a 4B model over a slow connection is not always possible on
# the day. So the provider is switchable: set LLM_PROVIDER=gemini and a
# GOOGLE_API_KEY, and the same agent, the same tools and the same prompts run
# against a larger hosted model. Nothing else in the pipeline changes.
#
# Set LLM_PROVIDER=ollama (or leave it unset) to go back.
# ---------------------------------------------------------------------------
PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


def model_label() -> str:
    """What to show in the UI, so it is never ambiguous what answered."""
    if PROVIDER == "gemini":
        return f"{GEMINI_MODEL} (Google AI Studio free tier)"
    return f"{MODEL} (local, via Ollama)"

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

# Added only when a class group chat has been uploaded, so the model is never
# told about a source that is not there.
CHAT_RULES = """
You also have the student's class WhatsApp group, which they uploaded.

8. For anything that would have been ANNOUNCED rather than printed - exam and
   viva dates, submission deadlines, room changes, what to bring tomorrow,
   cancelled lectures - call search_class_chat. Do not answer those from the
   transcript, and never from general knowledge.
9. When you answer from the group, say who said it and when. A date with no
   source is not useful to a student who has to act on it.
10. If the group does not contain the answer, say so. A wrong exam date is far
   worse than "I could not find it - check with your class representative".
"""


def get_llm():
    """The chat model for this run. See PROVIDER above."""
    if PROVIDER == "gemini":
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "LLM_PROVIDER=gemini but no GOOGLE_API_KEY (or GEMINI_API_KEY) is set. "
                "Get a free key at https://aistudio.google.com/apikey, or unset "
                "LLM_PROVIDER to run locally on Ollama."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:      # pragma: no cover
            raise RuntimeError(
                "LLM_PROVIDER=gemini needs the Google provider package. "
                "Run: pip install langchain-google-genai"
            ) from e
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL, temperature=0.1, google_api_key=key,
        )

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
    """Build the agent for the current session.

    Rebuild this after uploading or removing a class chat - the toolset and the
    system prompt both change with it, and an agent built before the upload has
    no way to know the chat exists.
    """
    tools = tools_for_session() + [make_retrieval_tool(retriever)]
    system = SYSTEM + (CHAT_RULES if has_class_chat() else "")
    kw = prompt_kwarg()
    if kw:
        return create_agent(get_llm(), tools, **{kw: system})
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


# ---------------------------------------------------------------------------
# Direct answering for class-group questions.
#
# Same decision as opening_summary above, for the same reason. A 1.7B model
# running on a laptop is not reliable at *choosing* to call a tool, and when it
# declines to call one it answers from its own weights instead - which for "when
# is the DBMS internal" means inventing a date. Running retrieval in Python and
# handing the model the results removes the choice: the messages are always
# fetched, and the model's only job is to read them back.
#
# The agent keeps the tool as well, so a larger model still gets to route for
# itself. This is a floor, not a ceiling.
# ---------------------------------------------------------------------------
ANNOUNCEMENT_QUESTION = re.compile(
    r"\b(exam|viva|practical|lab|submission|submit|deadline|due|assignment|project"
    r"|presentation|seminar|test|quiz|internal|syllabus|timetable|schedule"
    r"|reschedul|postpon|cancel|holiday|room|venue|hall|last date|drive|placement"
    r"|announce|said|told|group|when is|what did|which room|where is)\b",
    re.I,
)


def looks_like_class_question(question: str) -> bool:
    """True when a question is better answered from the group than the transcript."""
    return bool(ANNOUNCEMENT_QUESTION.search(question or ""))


def answer_from_chat(question: str) -> tuple[str, str]:
    """Retrieve from the class group in Python, then have the model read it back.

    Returns (raw_tool_output, plain_language_answer) so the UI can show the
    exact messages the answer was built from - the same "show your working"
    arrangement the study plan uses.
    """
    raw = search_class_chat.invoke({"query": question})

    try:
        llm = get_llm()
        out = llm.invoke([
            SystemMessage(content=(
                "Answer the student's question using ONLY the class group messages below. "
                "Say who said it and on what date. If several messages conflict, the LATEST "
                "one wins - a later message correcting a room or a date is the one that counts, "
                "and you should say that it was changed. If the messages do not answer the "
                "question, say so plainly and suggest asking the class representative. Never "
                "invent a date, a room or a deadline. Two to four sentences."
            )),
            HumanMessage(content=f"Question: {question}\n\n{raw}"),
        ])
        text = out.content if isinstance(out.content, str) else str(out.content)
        return raw, text.strip() or "See the messages below."
    except Exception as e:
        return raw, ("The model could not be reached, but the matching messages are shown "
                     "below exactly as they were sent. Details: " + str(e)[:150])


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
