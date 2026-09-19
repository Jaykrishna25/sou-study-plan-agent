"""
Transcript & Study Plan Agent - Streamlit app.

Built around the transcript upload, not a chat box. The moment a transcript
finishes ingesting, the study-plan tool runs and the weak areas, the prioritised
plan and the charts appear. Chat is underneath, for follow-ups.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src_edu.ingest import read_upload, build_index, get_retriever, WEAK_THRESHOLD, CRITICAL_THRESHOLD
from src_edu.classchat import load_class_chat, build_chat_index, get_chat_retriever
from src_edu.tools import (
    set_session, has_session, compute_plan, transcript_keywords,
    set_class_chat, clear_class_chat,
)
from src_edu.agent import (
    build_agent, ask, opening_summary, MODEL, model_label,
    answer_from_chat, looks_like_class_question,
)

st.set_page_config(page_title="Study Plan Agent", page_icon="=", layout="wide")

CSS = """
<style>
  #MainMenu, footer {visibility: hidden;}
  .block-container {padding-top: 2.2rem; max-width: 1180px;}
  h1, h2, h3 {letter-spacing: -0.02em;}
  .hero {font-size: 2.1rem; font-weight: 650; line-height: 1.15; margin-bottom: .3rem;}
  .sub {color: #8A94A6; font-size: .95rem; margin-bottom: 1.6rem;}
  .kpi {background: #151A24; border: 1px solid #222A38; border-radius: 14px;
        padding: 1rem 1.1rem; height: 100%;}
  .kpi .label {color: #8A94A6; font-size: .72rem; text-transform: uppercase;
               letter-spacing: .09em;}
  .kpi .value {font-size: 1.5rem; font-weight: 640; margin-top: .3rem;}
  .item {background: #141922; border: 1px solid #222A38; border-left: 3px solid #2E3A4D;
         border-radius: 12px; padding: .9rem 1rem; margin-bottom: .55rem;
         transition: transform .15s ease, border-color .15s ease;}
  .item:hover {transform: translateY(-2px); border-color: #35435A;}
  .item.critical {border-left-color: #E5484D;}
  .item.weak {border-left-color: #F5A524;}
  .item.strong {border-left-color: #2FBF71;}
  .item .nm {font-weight: 600;}
  .item .meta {color: #8A94A6; font-size: .8rem; margin-top: .25rem;}
  .chip {display:inline-block; background:#1B2230; border:1px solid #2A3547;
         border-radius:999px; padding:.18rem .6rem; font-size:.72rem; color:#AEB8C8;
         margin-right:.35rem;}
  .rank {display:inline-flex; align-items:center; justify-content:center;
         width:22px; height:22px; border-radius:6px; background:#1B2230;
         border:1px solid #2A3547; font-size:.72rem; margin-right:.5rem;}
  .box {border-radius: 12px; padding: .9rem 1.1rem; margin-bottom: .6rem;
        border: 1px solid; font-size: .9rem;}
  .box.bad  {background: rgba(229,72,77,.08);  border-color: rgba(229,72,77,.35);}
  .box.warn {background: rgba(245,165,36,.08); border-color: rgba(245,165,36,.35);}
  .box.ok   {background: rgba(47,191,113,.07); border-color: rgba(47,191,113,.3);}
  .src {color:#6F7A8C; font-size:.74rem; margin-top:.15rem;}
  .disc {color:#7A8496; font-size:.76rem; border-top:1px solid #1E2531;
         padding-top:.8rem; margin-top:1.6rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

for k, v in {
    "store": None, "retriever": None, "agent": None, "courses": None,
    "summary": "", "raw": "", "history": [], "filename": "", "threshold": WEAK_THRESHOLD,
    "chat_loaded": False, "chat_messages": 0, "chat_notes": [], "chat_name": "",
    "chat_announcements": [],
}.items():
    st.session_state.setdefault(k, v)


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Your transcript")
    up = st.file_uploader("Upload a transcript", type=["csv", "xlsx", "xls", "pdf", "txt"],
                          label_visibility="collapsed")
    use_sample = st.button("Use sample transcript", use_container_width=True)

    st.markdown("---")
    st.markdown("### Your class group")
    chat_up = st.file_uploader(
        "Upload an exported WhatsApp class group (.txt)", type=["txt"],
        label_visibility="collapsed", key="chat_upload",
    )
    st.caption(
        "Optional. Exam dates and deadlines are usually announced in the class group "
        "and nowhere else. Export from WhatsApp with **Without media**."
    )
    if st.session_state.chat_loaded:
        # No icon= here. Streamlit's icon argument takes an emoji, and a check
        # mark (U+2713) is a dingbat rather than an emoji, so it raises.
        st.success(f"{st.session_state.chat_messages} message(s) loaded")
        for note in st.session_state.chat_notes:
            st.caption("· " + note)
        if st.button("Remove the class group", use_container_width=True):
            clear_class_chat()
            st.session_state.update(
                chat_loaded=False, chat_messages=0, chat_notes=[], chat_name="",
                agent=build_agent(st.session_state.retriever) if st.session_state.retriever else None,
            )
            st.rerun()
    else:
        st.caption(
            "Phone numbers are removed before anything is indexed, and the file is "
            "never saved."
        )

    st.markdown("---")
    st.markdown("### Study settings")
    st.session_state.threshold = st.slider(
        "Flag a subject below this score", 40.0, 80.0, st.session_state.threshold, 2.5,
    )
    st.caption("Anything under this becomes a weak area in the plan. "
               f"Under {CRITICAL_THRESHOLD:g} is treated as at risk of failing.")

    st.markdown("---")
    st.caption(f"Model: `{model_label()}`")
    st.caption("Embeddings: `BAAI/bge-small-en-v1.5` · Vector store: Chroma")


def ingest(file_bytes: bytes, name: str):
    with st.status("Reading your transcript...", expanded=True) as s:
        st.write("Parsing subjects and marks")
        raw_text, courses = read_upload(file_bytes, name)
        if not courses:
            s.update(label="Could not read any subjects", state="error")
            st.error("No subjects were found. A transcript needs at least a subject column "
                     "and a marks or grade column.")
            return

        set_session(courses)
        st.write(f"Found {len(courses)} subjects. Building the search index...")
        store, n_chunks = build_index(raw_text, courses)
        retriever = get_retriever(store)

        st.write("Running the study plan tool")
        raw, summary = opening_summary(st.session_state.threshold)

        st.session_state.update(
            store=store, retriever=retriever, agent=build_agent(retriever),
            courses=courses, raw=raw, summary=summary, history=[], filename=name,
        )
        s.update(label=f"Ready - {len(courses)} subjects, {n_chunks} indexed chunks",
                 state="complete", expanded=False)


def ingest_chat(file_bytes: bytes, name: str):
    """Load and index the class group, then rebuild the agent around it.

    The agent has to be rebuilt rather than reused: search_class_chat is only
    bound when a chat exists, so an agent built before this upload cannot see
    it however the question is phrased.
    """
    with st.status("Reading the class group...", expanded=True) as s:
        try:
            st.write("Loading with WhatsAppChatLoader and removing phone numbers")
            result = load_class_chat(file_bytes, name)
            if not result.messages:
                s.update(label="No readable messages", state="error")
                st.error(
                    "No messages were found. Export the chat from WhatsApp using "
                    "**Without media** and upload the .txt file it produces."
                )
                return

            st.write(f"{len(result.messages)} messages over {result.days} day(s). Indexing...")
            store, n_docs = build_chat_index(result)
            set_class_chat(get_chat_retriever(store), f"{len(result.messages)} messages")

            st.session_state.update(
                chat_loaded=True,
                chat_messages=len(result.messages),
                chat_notes=result.notes,
                chat_name=name,
                chat_announcements=result.announcements[-8:],
                agent=build_agent(st.session_state.retriever) if st.session_state.retriever else None,
            )
            s.update(label=f"Class group ready - {n_docs} indexed documents",
                     state="complete", expanded=False)
        except Exception as e:
            s.update(label="Could not read the export", state="error")
            st.error(str(e)[:300])


if up is not None and up.name != st.session_state.filename:
    ingest(up.getvalue(), up.name)
elif use_sample:
    with open("sample/transcript.csv", "rb") as f:
        ingest(f.read(), "sample/transcript.csv")

if chat_up is not None and chat_up.name != st.session_state.chat_name:
    ingest_chat(chat_up.getvalue(), chat_up.name)


# ----------------------------------------------------------------- empty state
if not st.session_state.courses:
    st.markdown('<div class="hero">Know what to study next</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Upload your transcript. The weak areas and a prioritised '
                'study plan appear straight away - you do not have to know what to ask.</div>',
                unsafe_allow_html=True)
    a, b, c = st.columns(3)
    for col, (t, d) in zip((a, b, c), [
        ("Reads your own transcript", "CSV, Excel, PDF or text. Nothing is kept between sessions."),
        ("Ranks, never predicts", "Priorities come from Python arithmetic. No grade is forecast."),
        ("Matches jobs to what you studied", "Search keywords come from your transcript, not a box you type in."),
    ]):
        with col:
            st.markdown(f'<div class="kpi"><div class="label">{t}</div>'
                        f'<div class="meta" style="color:#8A94A6;font-size:.86rem;'
                        f'margin-top:.45rem">{d}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="disc">Guidance only. This is not an official academic record '
                'and it does not predict results.</div>', unsafe_allow_html=True)
    st.stop()


# ----------------------------------------------------------------- results
plan = compute_plan(st.session_state.threshold)
courses = st.session_state.courses

st.markdown('<div class="hero">Your study plan</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub">{st.session_state.filename} &middot; generated {plan.generated_at}</div>',
            unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
avg_colour = "#2FBF71" if plan.average_score >= 70 else ("#F5A524" if plan.average_score >= 55 else "#E5484D")
for col, label, value, colour in [
    (k1, "Subjects", str(len(courses)), "#E8ECF4"),
    (k2, "Average score", f"{plan.average_score:g}/100", avg_colour),
    (k3, "Weak areas", str(plan.weak_count), "#F5A524" if plan.weak_count else "#2FBF71"),
    (k4, "At risk of failing", str(plan.critical_count), "#E5484D" if plan.critical_count else "#2FBF71"),
]:
    with col:
        st.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                    f'<div class="value" style="color:{colour}">{value}</div></div>',
                    unsafe_allow_html=True)

st.write("")
st.markdown("#### What the plan says")
st.write(st.session_state.summary)

if plan.critical_count:
    st.markdown(f'<div class="box bad"><b>{plan.critical_count} subject(s) at risk of failing</b><br>'
                f'Scoring under {CRITICAL_THRESHOLD:g}/100. These are first in the plan below.</div>',
                unsafe_allow_html=True)
elif plan.weak_count:
    st.markdown(f'<div class="box warn"><b>{plan.weak_count} weak subject(s)</b><br>'
                f'Below your threshold of {st.session_state.threshold:g}/100, but none at failing level.</div>',
                unsafe_allow_html=True)
else:
    st.markdown(f'<div class="box ok">No subject falls below {st.session_state.threshold:g}/100. '
                f'Nothing is flagged for revision.</div>', unsafe_allow_html=True)

# ---------------- charts ----------------
st.write("")
c1, c2 = st.columns([1.25, 1])

with c1:
    st.markdown("##### Every subject, weakest first")
    ordered = sorted(courses, key=lambda c: c.total)
    colours = ["#E5484D" if c.total < CRITICAL_THRESHOLD
               else ("#F5A524" if c.total < st.session_state.threshold else "#2FBF71")
               for c in ordered]
    fig = go.Figure(go.Bar(
        x=[c.total for c in ordered],
        y=[c.subject for c in ordered],
        orientation="h", marker_color=colours,
        hovertemplate="%{y}: %{x}/100<extra></extra>",
    ))
    fig.add_vline(x=st.session_state.threshold, line_dash="dash", line_color="#8A94A6")
    fig.update_layout(
        height=max(320, 26 * len(ordered)), margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="score out of 100", range=[0, 100], gridcolor="#1C2430"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="src">Dashed line is your threshold. Red is at risk of failing.</div>',
                unsafe_allow_html=True)

with c2:
    st.markdown("##### Where your study hours go")
    if plan.items:
        fig2 = go.Figure(go.Pie(
            labels=[i.subject for i in plan.items],
            values=[i.hours_per_week for i in plan.items],
            hole=.62,
            marker=dict(colors=["#E5484D" if i.severity == "critical" else "#F5A524" for i in plan.items],
                        line=dict(color="#0C0F16", width=2)),
            textinfo="label+value", textfont=dict(size=11),
        ))
        fig2.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10, l=10, r=10),
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f'<div class="src">{plan.total_hours} hours per week in total, '
                    f'weighted by how far below the line each subject is.</div>',
                    unsafe_allow_html=True)
    else:
        st.info("No revision hours allocated - nothing is below your threshold.")

# ---------------- the plan ----------------
st.write("")
st.markdown("##### Prioritised plan")
if not plan.items:
    st.markdown('<div class="box ok">Nothing to revise at this threshold. '
                'Lower the slider to see where you are closest to the line.</div>',
                unsafe_allow_html=True)
for i in plan.items:
    st.markdown(
        f'<div class="item {i.severity}"><div class="nm">'
        f'<span class="rank">{i.priority}</span>{i.subject} '
        f'<span class="chip">{i.code}</span><span class="chip">sem {i.semester}</span>'
        f'<span class="chip">{i.hours_per_week} hrs/week</span></div>'
        f'<div class="meta">Scored {i.score:g}/100 (grade {i.grade}) &middot; '
        f'{i.credits:g} credits &middot; {i.reason}</div></div>',
        unsafe_allow_html=True)

if plan.strong:
    st.markdown("##### Strongest subjects")
    for s in plan.strong:
        st.markdown(f'<div class="item strong"><div class="nm">{s}</div>'
                    f'<div class="meta">Used to match internships to what you actually studied.</div>'
                    f'</div>', unsafe_allow_html=True)
    st.markdown('<div class="src">Search keywords drawn from these: '
                + ", ".join(transcript_keywords()) + '</div>', unsafe_allow_html=True)

with st.expander("Raw study plan tool output (what the model was given)"):
    st.code(st.session_state.raw, language="text")

# ----------------------------------------------------------------- chat
st.write("")
st.markdown("#### Ask about your transcript")
st.caption("Answers come from your transcript and a live listings feed. "
           "The adviser is told never to calculate a figure or predict a grade.")

cols = st.columns(4)
suggestions = [
    "What should I revise first?",
    "What grade did I get in databases?",
    "Find internships that match my transcript",
    "Which subjects am I strongest in?",
]
clicked = None
for col, s in zip(cols, suggestions):
    if col.button(s, use_container_width=True):
        clicked = s

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

typed = st.chat_input("Ask a follow-up question about your transcript")
question = clicked or typed

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    # A question about an announcement goes straight to the class group rather
    # than through the agent's own routing. See answer_from_chat for why: a
    # small local model that declines to call the tool answers from its weights
    # instead, and an invented exam date is the worst output this app can give.
    use_chat = st.session_state.chat_loaded and looks_like_class_question(question)

    with st.chat_message("assistant"):
        if use_chat:
            with st.spinner("Searching your class group..."):
                raw_msgs, answer = answer_from_chat(question)
            st.markdown(answer)
            with st.expander("The exact messages this came from"):
                st.text(raw_msgs)
        else:
            with st.spinner("Checking your transcript..."):
                answer = ask(st.session_state.agent, question)
            st.markdown(answer)

    st.session_state.history.append(("assistant", answer))

st.markdown('<div class="disc">Guidance only, from the transcript you uploaded. It does not '
            'predict results and is not an official academic record. Listings come from a live '
            'public jobs feed and are not endorsements.</div>', unsafe_allow_html=True)
