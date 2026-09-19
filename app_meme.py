"""
Meme & Internet Culture Historian - Streamlit app.

Built around one clear action: compile today's digest. The chat is secondary,
for one-off lookups. The brief is explicit that a digest must be a single
proactive action across several trends, not a chat box in disguise.
"""
from __future__ import annotations

import streamlit as st

from src_meme.ingest import load_entries, build_index, get_retriever
from src_meme.tools import set_retriever, lookup_meme_history
from src_meme.agent import build_agent, ask, compile_digest, MODEL

st.set_page_config(page_title="Culture Desk", page_icon="*", layout="wide")

CSS = """
<style>
  #MainMenu, footer {visibility: hidden;}
  .block-container {padding-top: 2rem; max-width: 1100px;}
  h1, h2, h3 {letter-spacing: -0.02em;}
  .hero {font-size: 2.3rem; font-weight: 680; line-height: 1.1;
         background: linear-gradient(92deg,#8B5CF6,#EC4899 55%,#F59E0B);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         margin-bottom: .25rem;}
  .sub {color: #8A94A6; font-size: .95rem; margin-bottom: 1.4rem;}
  .card {background:#151A24; border:1px solid #242C3B; border-radius:14px;
         padding:1rem 1.1rem; margin-bottom:.6rem;}
  .card:hover {border-color:#364054;}
  .trend {font-weight:600; line-height:1.35;}
  .meta {color:#8A94A6; font-size:.78rem; margin-top:.3rem;}
  .chip {display:inline-block; background:#1B2230; border:1px solid #2A3547;
         border-radius:999px; padding:.15rem .6rem; font-size:.7rem;
         color:#AEB8C8; margin-right:.3rem;}
  .chip.live {border-color:#EC4899; color:#F9A8D4; background:rgba(236,72,153,.1);}
  .chip.hist {border-color:#8B5CF6; color:#C4B5FD; background:rgba(139,92,246,.1);}
  .digest {background:#12161F; border:1px solid #242C3B; border-left:3px solid #8B5CF6;
           border-radius:14px; padding:1.2rem 1.3rem; line-height:1.65;}
  .disc {color:#7A8496; font-size:.76rem; border-top:1px solid #1E2531;
         padding-top:.8rem; margin-top:1.6rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

for k, v in {
    "store": None, "retriever": None, "agent": None, "entries": None,
    "digest": "", "topics": [], "raw": "", "history": [],
}.items():
    st.session_state.setdefault(k, v)


@st.cache_resource(show_spinner=False)
def boot():
    """Load the corpus and build the index once per process."""
    entries = load_entries()
    store, n_chunks = build_index(entries)
    return entries, store, get_retriever(store), n_chunks


with st.sidebar:
    st.markdown("### Culture desk")
    st.caption("A historian that knows where things came from, and can see what "
               "is happening right now.")
    st.markdown("---")
    n_topics = st.slider("Trends per digest", 3, 8, 5)
    st.caption("Each one is cross-referenced against the reference set "
               "individually, so a bigger digest means more lookups.")
    st.markdown("---")
    st.caption(f"Model: `{MODEL}` running locally via Ollama")
    st.caption("Embeddings: `BAAI/bge-small-en-v1.5` · Vector store: Chroma")

# ---- boot the index ----
if st.session_state.retriever is None:
    with st.status("Loading the reference set...", expanded=True) as s:
        st.write("Reading entries, one per section")
        entries, store, retriever, n_chunks = boot()
        set_retriever(retriever)
        st.write(f"{len(entries)} entries, {n_chunks} indexed chunks")
        st.write("Starting the historian")
        st.session_state.update(
            entries=entries, store=store, retriever=retriever, agent=build_agent(),
        )
        s.update(label=f"Ready - {len(entries)} entries indexed",
                 state="complete", expanded=False)

st.markdown('<div class="hero">Today on the internet</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">One button. It pulls what is trending right now, looks each '
            'one up in its own history, and tells you which are genuinely new.</div>',
            unsafe_allow_html=True)

c1, c2 = st.columns([1, 3])
with c1:
    go = st.button("Compile today's digest", type="primary", use_container_width=True)
with c2:
    st.caption(f"Fetches {n_topics} live trends and runs {n_topics} separate history "
               "lookups in one action.")

if go:
    with st.status("Compiling...", expanded=True) as s:
        st.write("Fetching live trending topics")
        topics, raw, digest = compile_digest(n_topics)
        st.write(f"Cross-referencing {len(topics)} topic(s) against the reference set")
        st.write("Writing the digest")
        st.session_state.update(topics=topics, raw=raw, digest=digest)
        s.update(label=f"Digest ready - {len(topics)} trends cross-referenced",
                 state="complete", expanded=False)

if st.session_state.digest:
    st.write("")
    st.markdown(f'<div class="digest">{st.session_state.digest}</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("##### What it was built from")
    for t in st.session_state.topics:
        found = "ENTRY:" in (t.get("history") or "")
        st.markdown(
            f'<div class="card"><div class="trend">{t["title"]}</div>'
            f'<div class="meta">'
            f'<span class="chip live">live · {t["where"]}</span>'
            f'<span class="chip hist">{"history found" if found else "no match in reference set"}</span>'
            f'score {t["score"]}</div></div>',
            unsafe_allow_html=True)

    with st.expander("Raw material the digest was written from"):
        st.code(st.session_state.raw, language="text")
else:
    st.write("")
    a, b, c = st.columns(3)
    for col, (t, d) in zip((a, b, c), [
        ("Two tools, kept apart",
         "One knows history, one knows today. Asking what something means never touches the live feed."),
        ("Fuzzy names work",
         'The reference set records what people actually call things, so "that skibidi thing" finds the entry.'),
        ("It admits when it is new",
         "A forced historical connection is worse than saying a trend has no precedent."),
    ]):
        with col:
            st.markdown(f'<div class="card"><div class="trend">{t}</div>'
                        f'<div class="meta">{d}</div></div>', unsafe_allow_html=True)

# ---------------- lookup chat ----------------
st.write("")
st.markdown("#### Look something up")
st.caption("Separate from the digest. This searches the reference set — try a vague name.")

cols = st.columns(4)
suggestions = ["that skibidi thing", "the guy looking back",
               "why is Pepe controversial", "what is rizz"]
clicked = None
for col, s in zip(cols, suggestions):
    if col.button(s, use_container_width=True):
        clicked = s

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

typed = st.chat_input("Ask about any meme, trend or piece of slang")
question = clicked or typed

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking the archive..."):
            answer = ask(st.session_state.agent, question)
        st.markdown(answer)
    st.session_state.history.append(("assistant", answer))

st.markdown('<div class="disc">Trending data comes from a live public feed and reflects one '
            'slice of the internet, not all of it. Historical entries are a curated reference '
            'set written for this project; where it has no entry, the historian says so rather '
            'than guessing.</div>', unsafe_allow_html=True)
