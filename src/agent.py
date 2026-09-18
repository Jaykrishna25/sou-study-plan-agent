"""
Agent: ChatOllama + tool calling, ReAct style.

Design decision worth defending: the plain-language summary shown on upload is
produced by running the analysis tool DIRECTLY and then asking the model to
phrase the result. The tool genuinely runs automatically the moment ingestion
finishes - it does not depend on a small local model choosing to call it. The
numbers are therefore always correct; the model only ever rewords them.

Follow-up questions go through the agent proper, which picks between searching
the statement, re-running the analysis, and fetching live prices.
"""
from __future__ import annotations

import inspect

import os

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from src.tools import TOOLS, analyze_portfolio

# create_agent moved between packages across versions; support both.
try:
    from langchain.agents import create_agent           # LangChain 1.x
except ImportError:                                      # pragma: no cover
    from langgraph.prebuilt import create_react_agent as create_agent

# qwen3:4b rather than 0.6b. The smaller model was measurably unreliable at tool
# calling: it emitted a raw <search_statement query="..."> string as its answer
# instead of invoking the tool, reported two different holdings as "100% weight",
# and refused a question the analysis tool could answer. Tool-call reliability is
# the whole product here, so the larger model earns its extra memory.
# Override with OLLAMA_MODEL if a machine cannot spare the RAM.
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

SYSTEM = """You are a portfolio assistant working with ONE statement the user just uploaded.

HARD RULES
1. You never calculate. The tools calculate. If you need a number, call a tool.
   Never estimate, round differently, or infer a figure the tools did not return.
2. Always say where a number came from: "from your statement" or "live market price".
3. Never recommend buying, selling or holding anything. You explain, you do not advise.
4. If a tool reports it could not find or price something, say so plainly.
   Do not substitute a guess.
5. Keep answers short and in plain language. Assume the user is not a finance professional.

TOOLS
- search_statement: find what the uploaded statement says about a holding or section.
- analyze_portfolio: recompute gains, losses and concentration from the statement.
- get_market_data: current price and day change for one ticker symbol.

This is informational only and not personalised financial advice."""


def get_llm():
    return ChatOllama(model=MODEL, temperature=0.1)


def make_retrieval_tool(retriever):
    """Wraps this session's retriever so the agent can quote the statement."""

    @tool
    def search_statement(query: str) -> str:
        """Search the portfolio statement the user uploaded in this session.

        Use this for anything about what the statement itself says - a holding's
        quantity, purchase price, purchase date or sector. Returns the matching
        sections verbatim. This is STATEMENT data, not live market data.
        """
        try:
            docs = retriever.invoke(query)
        except Exception as e:
            return f"Could not search the statement: {str(e)[:120]}"
        if not docs:
            return "Nothing in the uploaded statement matches that. Do not guess an answer."
        return "\n\n---\n\n".join(d.page_content for d in docs)

    return search_statement


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
    """Build the tool-calling agent for this uploaded statement.

    The system-prompt parameter has been renamed several times across
    LangChain and LangGraph releases, so the name is discovered from the
    signature at runtime instead of being hard-coded.
    """
    tools = TOOLS + [make_retrieval_tool(retriever)]
    kw = prompt_kwarg()
    if kw:
        return create_agent(get_llm(), tools, **{kw: SYSTEM})
    return create_agent(get_llm(), tools)

def ask(agent, question: str, history: list | None = None) -> str:
    """Run one turn. Returns the final text, never raises into the UI."""
    messages = list(history or []) + [HumanMessage(content=question)]
    try:
        result = agent.invoke({"messages": messages})
        msgs = result.get("messages", [])
        for m in reversed(msgs):
            text = getattr(m, "content", "")
            if text and getattr(m, "type", "") in ("ai", "AIMessageChunk"):
                return text if isinstance(text, str) else str(text)
        return "The assistant did not return an answer. Try rephrasing the question."
    except Exception as e:
        return f"The assistant could not complete that request: {str(e)[:200]}"


def opening_summary() -> tuple[str, str]:
    """
    Runs on ingestion. Returns (raw_tool_output, plain_language_summary).

    The tool output is the source of truth and is shown in the UI as structured
    cards; the summary is the model's rewording of it.
    """
    raw = analyze_portfolio.invoke({"concentration_threshold_pct": 25.0})
    try:
        llm = get_llm()
        reply = llm.invoke([
            SystemMessage(content=SYSTEM),
            HumanMessage(content=(
                "Below is the output of the portfolio analysis tool for the statement "
                "just uploaded. Rewrite it as a short plain-language briefing of four to "
                "six sentences for someone who is not a finance professional. Lead with "
                "overall performance, then any concentration or underperformance flags. "
                "Use only the numbers given. Do not add advice.\n\n" + raw
            )),
        ])
        summary = reply.content if isinstance(reply.content, str) else str(reply.content)
    except Exception as e:
        summary = ("Analysis completed. (The language model is unavailable, so the "
                   f"structured results below are shown without a written summary: {str(e)[:120]})")
    return raw, summary.strip()
