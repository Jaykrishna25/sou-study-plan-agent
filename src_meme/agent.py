"""
Agent: ChatOllama + tool calling, ReAct style, plus a proactive digest.

Design decision worth defending: `compile_digest` does not ask the model to
"go and fetch some trends". It fetches several live topics itself, retrieves
historical context for EACH ONE, and only then asks the model to write the
digest from that assembled material.

That is the difference the brief is testing. An agent told to compile a digest
will, left to itself, fetch one topic, answer about it, and stop - one Q&A pair
wearing a digest's clothes. Doing the fan-out in Python guarantees several
trends are genuinely cross-referenced in a single action, and makes the result
the same every time.

Follow-up lookups still go through the agent proper, which chooses between the
history tool and the live feed.
"""
from __future__ import annotations

import inspect
import os

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src_meme.tools import TOOLS, fetch_trending, lookup_meme_history

try:
    from langchain.agents import create_agent           # LangChain 1.x
except ImportError:                                      # pragma: no cover
    from langgraph.prebuilt import create_react_agent as create_agent

MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

SYSTEM = """You are an internet culture historian. You know meme history and you can see
what is trending right now, and you keep those two things clearly apart.

HARD RULES
1. Two tools, two jobs. lookup_meme_history is HISTORY - origins and meanings.
   get_trending_now is TODAY - live headlines. Never answer a "what does this
   mean" question from the live feed, and never claim something is trending
   because it appears in the reference set.
2. Only state an origin, a date or a creator that came back from the history
   tool. If the reference set does not have it, say so - do not fill the gap
   from memory.
3. When you connect a live trend to meme history, say what the connection is.
   If there is no honest connection, say the trend is genuinely new. A forced
   link is worse than admitting there isn't one.
4. Be entertaining but accurate. This is a culture desk, not an encyclopedia
   and not a comedy act.
5. Keep answers short - three to six sentences unless asked for more.
"""

DIGEST_SYSTEM = """You are writing today's internet culture digest.

Below are live trending topics, each paired with whatever the historical
reference set returned for it. For EACH topic, write two or three sentences:
what it is, and either the honest historical connection or a plain statement
that it appears to be genuinely new.

RULES
- Use ONLY the material given. Do not add origins, dates or creators from your
  own memory.
- Do not force a connection. "This one looks genuinely new" is a good answer and
  you should use it when the retrieved history does not actually relate.
- Keep each item tight. Open with one line setting the tone for the day overall.
- Do not number the items; write it as a short piece of prose with the topic
  names in bold.
"""


def get_llm() -> ChatOllama:
    return ChatOllama(model=MODEL, temperature=0.3)


_PROMPT_KW = None


def prompt_kwarg() -> str:
    global _PROMPT_KW
    if _PROMPT_KW is None:
        params = inspect.signature(create_agent).parameters
        _PROMPT_KW = next(
            (k for k in ("system_prompt", "prompt", "state_modifier", "messages_modifier")
             if k in params),
            "",
        )
    return _PROMPT_KW


def build_agent():
    kw = prompt_kwarg()
    if kw:
        return create_agent(get_llm(), TOOLS, **{kw: SYSTEM})
    return create_agent(get_llm(), TOOLS)


def ask(agent, question: str, history: list | None = None) -> str:
    messages = list(history or []) + [HumanMessage(content=question)]
    try:
        result = agent.invoke({"messages": messages})
    except Exception as e:
        return "The historian could not be reached just now. Details: " + str(e)[:200]

    msgs = result.get("messages", []) if isinstance(result, dict) else []
    for m in reversed(msgs):
        text = getattr(m, "content", "")
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        if text and getattr(m, "type", "") in ("ai", "AIMessageChunk"):
            return str(text).strip()
    return "No answer came back. Try rephrasing."


def compile_digest(n_topics: int = 5) -> tuple[list[dict], str, str]:
    """Fetch several live trends, cross-reference each, and write the digest.

    Returns (topics, raw_material, digest_text). The raw material is shown in the
    UI so the digest can be checked against exactly what it was built from.
    """
    topics, err = fetch_trending(n_topics)
    if not topics:
        return [], "", ("Today's digest could not be compiled: the live trending feed "
                        "could not be reached. " + (err or ""))

    # Fan out: one history lookup per topic, in Python, every time.
    blocks: list[str] = []
    for t in topics:
        history = lookup_meme_history.invoke({"query": t["title"]})
        t["history"] = history
        blocks.append(
            f"TRENDING: {t['title']}  (on {t['where']}, score {t['score']})\n"
            f"RETRIEVED HISTORY:\n{history}"
        )
    raw = "\n\n========\n\n".join(blocks)

    try:
        llm = get_llm()
        out = llm.invoke([
            SystemMessage(content=DIGEST_SYSTEM),
            HumanMessage(content=raw),
        ])
        text = out.content if isinstance(out.content, str) else str(out.content)
        return topics, raw, text.strip() or "The digest came back empty."
    except Exception as e:
        return topics, raw, (
            "The write-up could not be generated (" + str(e)[:120] + "), but the "
            "trends and their retrieved history are below - the fetch and the "
            "cross-reference both ran."
        )
