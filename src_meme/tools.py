"""
The two required tools for the Wildcard track.

  1. get_trending_now   - RIGHT NOW data from a live public feed.
  2. lookup_meme_history - HISTORICAL data from the curated corpus.

The docstrings deliberately over-emphasise the difference. The failure mode for
this agent is confusing "what does this mean" with "what is trending today", and
the only thing standing between it and that mistake is how these two tools are
described.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from langchain_core.tools import tool

# Hacker News: free, no key, no rate limit worth worrying about. The brief
# explicitly allows a news-headlines API as a proxy for a trending feed.
HN_TOP = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM = "https://hacker-news.firebaseio.com/v0/item/{}.json"

# Reddit's public listing works without auth when a real User-Agent is sent.
REDDIT_POPULAR = "https://www.reddit.com/r/popular/hot.json?limit=15"

TIMEOUT = 12
UA = {"User-Agent": "meme-historian/1.0 (hackathon project)"}

_RETRIEVER = None


def set_retriever(retriever) -> None:
    global _RETRIEVER
    _RETRIEVER = retriever


def _get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def fetch_trending(limit: int = 6) -> tuple[list[dict], str | None]:
    """Return (topics, error). Tries Reddit first, falls back to Hacker News."""
    try:
        data = _get_json(REDDIT_POPULAR)
        children = data.get("data", {}).get("children", [])
        topics = [{
            "title": c["data"].get("title", "").strip(),
            "where": "r/" + c["data"].get("subreddit", "?"),
            "score": c["data"].get("score", 0),
            "source": "Reddit r/popular",
        } for c in children if c.get("data", {}).get("title")]
        if topics:
            return topics[:limit], None
    except Exception as e:
        reddit_err = str(e)[:120]
    else:
        reddit_err = "no items returned"

    try:
        ids = _get_json(HN_TOP)[:limit]
        topics = []
        for i in ids:
            item = _get_json(HN_ITEM.format(i))
            if item and item.get("title"):
                topics.append({
                    "title": item["title"].strip(),
                    "where": "Hacker News",
                    "score": item.get("score", 0),
                    "source": "Hacker News front page",
                })
        if topics:
            return topics, None
        return [], f"Reddit failed ({reddit_err}); Hacker News returned nothing."
    except Exception as e:
        return [], f"Reddit failed ({reddit_err}); Hacker News failed ({str(e)[:120]})."


@tool
def get_trending_now(limit: int = 6) -> str:
    """Fetch what is trending on the internet RIGHT NOW, from a live public feed.

    This is TODAY'S data. It is NOT historical and it says nothing about what a
    meme means or where it came from. Use this only when the question is about
    what is currently popular, currently being discussed, or trending today.

    For what something MEANS or where it CAME FROM, use lookup_meme_history
    instead. Returns live headlines with where each one is trending.
    """
    topics, err = fetch_trending(limit)
    if err and not topics:
        return ("The live trending feed could not be reached, so there is nothing "
                "current to report. Details: " + err)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"LIVE TRENDING TOPICS (fetched {stamp}) - this is RIGHT NOW data:", ""]
    for i, t in enumerate(topics, 1):
        lines.append(f"{i}. {t['title']}  [{t['where']}, score {t['score']}]")
    lines.append("")
    lines.append("Source: " + topics[0]["source"] + ". These are live headlines, "
                 "not entries from the meme reference set.")
    return "\n".join(lines)


@tool
def lookup_meme_history(query: str) -> str:
    """Look up the HISTORY of a meme, trend or piece of internet slang.

    This searches a curated reference set of internet culture: origin, first
    appearance, what it means, how it varied, and why it mattered. This is
    HISTORICAL knowledge. It knows nothing about what is trending today.

    The query can be vague or misspelled - the reference set records the slang
    names people actually use, so "that skibidi thing" or "the guy looking back"
    will find the right entry. Use this for any question about what something
    means or where it came from.
    """
    if _RETRIEVER is None:
        return "The reference set has not been loaded yet."
    try:
        docs = _RETRIEVER.invoke(query)
    except Exception as e:
        return "The history lookup failed: " + str(e)[:150]
    if not docs:
        return (f"Nothing in the reference set matches '{query}'. "
                "It may be too new to be recorded, or too niche.")
    return "\n\n---\n\n".join(
        f"ENTRY: {d.metadata.get('name', 'unknown')}\n{d.page_content}" for d in docs
    )


TOOLS = [get_trending_now, lookup_meme_history]
