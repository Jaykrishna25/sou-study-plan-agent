"""
Portfolio & Statement Simplifier Agent - Streamlit app.

The interface is built around the upload, not a chat box. The moment a
statement finishes ingesting, the analysis tool runs and the summary, risk
flags and charts appear. Chat is available underneath for follow-ups.
"""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ingest import read_upload, build_index, get_retriever
from src.tools import set_session, fetch_prices, compute_analysis
from src.agent import build_agent, ask, opening_summary, MODEL

st.set_page_config(page_title="Portfolio Simplifier", page_icon="[]", layout="wide")

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
  .pos {background: #141922; border: 1px solid #222A38; border-left: 3px solid #2E3A4D;
        border-radius: 12px; padding: .85rem 1rem; margin-bottom: .55rem;
        transition: transform .15s ease, border-color .15s ease;}
  .pos:hover {transform: translateY(-2px); border-color: #35435A;}
  .pos.up {border-left-color: #2FBF71;}
  .pos.down {border-left-color: #E5484D;}
  .pos.flag {border-left-color: #F5A524;}
  .pos .nm {font-weight: 600;}
  .pos .meta {color: #8A94A6; font-size: .8rem; margin-top: .2rem;}
  .chip {display:inline-block; background:#1B2230; border:1px solid #2A3547;
         border-radius:999px; padding:.18rem .6rem; font-size:.72rem; color:#AEB8C8;
         margin-right:.35rem;}
  .flagbox {border-radius: 12px; padding: .9rem 1.1rem; margin-bottom: .6rem;
            border: 1px solid; font-size: .9rem;}
  .flagbox.warn {background: rgba(245,165,36,.08); border-color: rgba(245,165,36,.35);}
  .flagbox.bad  {background: rgba(229,72,77,.08);  border-color: rgba(229,72,77,.35);}
  .flagbox.ok   {background: rgba(47,191,113,.07); border-color: rgba(47,191,113,.3);}
  .src {color:#6F7A8C; font-size:.74rem; margin-top:.15rem;}
  .disc {color:#7A8496; font-size:.76rem; border-top:1px solid #1E2531;
         padding-top:.8rem; margin-top:1.6rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

for k, v in {
    "store": None, "retriever": None, "agent": None,
    "analysis": None, "summary": "", "raw": "", "history": [],
    "filename": "", "threshold": 25.0,
}.items():
    st.session_state.setdefault(k, v)


def money(x): return "-" if x is None else f"INR {x:,.0f}"


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Your statement")
    up = st.file_uploader("Upload a portfolio statement", type=["csv", "xlsx", "xls", "pdf"],
                          label_visibility="collapsed")
    use_sample = st.button("Use sample statement", use_container_width=True)

    st.markdown("---")
    st.markdown("### Risk settings")
    st.session_state.threshold = st.slider(
        "Concentration flag threshold (% of portfolio in one holding)",
        5.0, 50.0, st.session_state.threshold, 2.5,
    )
    st.caption("Any single position above this share of portfolio value is flagged.")

    st.markdown("---")
    st.caption(f"Model: `{MODEL}` running locally via Ollama")
    st.caption("Embeddings: `BAAI/bge-small-en-v1.5` - Vector store: Chroma")


def ingest(file_bytes: bytes, name: str):
    with st.status("Reading your statement...", expanded=True) as s:
        st.write("Parsing holdings")
        raw_text, holdings = read_upload(file_bytes, name)
        if not holdings:
            s.update(label="Could not read any holdings", state="error")
            st.error("No holdings were found. A CSV needs columns for symbol, quantity "
                     "and purchase price.")
            return

        st.write(f"Found {len(holdings)} holdings. Fetching live prices...")
        prices, errors = fetch_prices([h.symbol for h in holdings])
        # The market feed sometimes returns NaN or zero rather than omitting a
        # symbol. Treat those as "no price available" instead of letting a NaN
        # propagate through every total in the analysis.
        clean, bad = {}, dict(errors)
        for sym, raw_px in prices.items():
            try:
                px = float(raw_px)
            except (TypeError, ValueError):
                px = float("nan")
            if math.isfinite(px) and px > 0:
                clean[sym] = px
            else:
                bad[sym] = "market feed returned no usable price"
        set_session(holdings, clean, bad)

        st.write("Building the search index for this statement")
        store, n_chunks = build_index(raw_text, holdings)
        retriever = get_retriever(store)

        st.write("Running the analysis tool")
        raw, summary = opening_summary()

        st.session_state.update(
            store=store, retriever=retriever, agent=build_agent(retriever),
            analysis=compute_analysis(st.session_state.threshold),
            raw=raw, summary=summary, history=[], filename=name,
        )
        s.update(label=f"Ready - {len(holdings)} holdings, {n_chunks} indexed chunks",
                 state="complete", expanded=False)


if up is not None and up.name != st.session_state.filename:
    ingest(up.getvalue(), up.name)
elif use_sample:
    with open("sample/portfolio.csv", "rb") as f:
        ingest(f.read(), "sample/portfolio.csv")


# ----------------------------------------------------------------- empty state
if st.session_state.analysis is None:
    st.markdown('<div class="hero">Understand your portfolio in plain language</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub">Upload a brokerage statement or holdings sheet. '
                'The analysis runs the moment it is read - no question needed.</div>',
                unsafe_allow_html=True)
    a, b, c = st.columns(3)
    for col, (t, d) in zip((a, b, c), [
        ("Reads your own statement", "CSV, Excel or PDF. Nothing is shared between sessions."),
        ("Computes, never guesses", "Gains, losses and concentration are Python arithmetic, not model output."),
        ("Says where numbers came from", "Every figure is labelled as your statement or a live market price."),
    ]):
        with col:
            st.markdown(f'<div class="kpi"><div class="label">{t}</div>'
                        f'<div class="meta" style="color:#8A94A6;font-size:.86rem;'
                        f'margin-top:.45rem">{d}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="disc">Informational only. Not personalised financial advice.</div>',
                unsafe_allow_html=True)
    st.stop()


# ----------------------------------------------------------------- results
a = compute_analysis(st.session_state.threshold)
st.session_state.analysis = a

st.markdown('<div class="hero">Your portfolio, explained</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub">{st.session_state.filename} &middot; analysed {a.generated_at}</div>',
            unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
gain_colour = "#2FBF71" if a.total_gain >= 0 else "#E5484D"
for col, label, value, colour in [
    (k1, "Invested (statement)", money(a.total_invested), "#E8ECF4"),
    (k2, "Current value (live)", money(a.total_value), "#E8ECF4"),
    (k3, "Gain / loss", f"{money(a.total_gain)} ({a.total_gain_percent:+.2f}%)", gain_colour),
    (k4, "Holdings", str(len(a.positions)), "#E8ECF4"),
]:
    with col:
        st.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                    f'<div class="value" style="color:{colour}">{value}</div></div>',
                    unsafe_allow_html=True)

st.write("")
st.markdown("#### What the analysis found")
st.write(st.session_state.summary)

# flags
if a.concentration_flags:
    for f in a.concentration_flags:
        st.markdown(f'<div class="flagbox warn"><b>Concentration risk</b><br>{f}</div>',
                    unsafe_allow_html=True)
else:
    st.markdown(f'<div class="flagbox ok">No position exceeds the '
                f'{a.threshold:.0f}% concentration threshold.</div>', unsafe_allow_html=True)
for f in a.loss_flags:
    st.markdown(f'<div class="flagbox bad"><b>Underperforming</b><br>{f}</div>',
                unsafe_allow_html=True)
if a.unpriced:
    st.markdown(f'<div class="flagbox warn"><b>No live price</b><br>'
                f'{", ".join(a.unpriced)} could not be priced and are excluded from '
                f'value and weight figures.</div>', unsafe_allow_html=True)

# charts
st.write("")
c1, c2 = st.columns([1, 1.25])
priced = [p for p in a.positions if p.priced and p.current_value is not None]

with c1:
    st.markdown("##### Concentration by holding")
    if priced:
        colours = ["#F5A524" if (p.weight_percent or 0) > a.threshold else "#4F8DF7" for p in priced]
        fig = go.Figure(go.Pie(
            labels=[p.symbol.replace(".NS", "") for p in priced],
            values=[p.current_value for p in priced],
            hole=.62, marker=dict(colors=colours, line=dict(color="#0C0F16", width=2)),
            textinfo="label+percent", textfont=dict(size=11),
        ))
        fig.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10, l=10, r=10),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="src">Amber = above your flag threshold. '
                    'Source: statement quantities x live prices.</div>', unsafe_allow_html=True)
    else:
        st.info("No holding could be priced, so concentration cannot be charted.")

with c2:
    st.markdown("##### Gain / loss by holding")
    if priced:
        ordered = sorted(priced, key=lambda p: p.gain_percent or 0)
        fig2 = go.Figure(go.Bar(
            x=[p.gain_percent or 0 for p in ordered],
            y=[p.symbol.replace(".NS", "") for p in ordered],
            orientation="h",
            marker_color=["#E5484D" if (p.gain_percent or 0) < 0 else "#2FBF71" for p in ordered],
            hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
        ))
        fig2.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(title="% against purchase price", gridcolor="#1C2430"),
                           yaxis=dict(gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="src">Percentage change against the purchase price in your statement.</div>',
                    unsafe_allow_html=True)

# positions
st.write("")
st.markdown("##### Holdings")
for p in a.positions:
    if not p.priced:
        st.markdown(f'<div class="pos"><div class="nm">{p.name} '
                    f'<span class="chip">{p.symbol}</span></div>'
                    f'<div class="meta">{p.quantity:g} shares &middot; invested {money(p.invested)} '
                    f'&middot; no live price available</div></div>', unsafe_allow_html=True)
        continue
    cls = "flag" if (p.weight_percent or 0) > a.threshold else ("up" if (p.gain_percent or 0) >= 0 else "down")
    st.markdown(
        f'<div class="pos {cls}"><div class="nm">{p.name} '
        f'<span class="chip">{p.symbol}</span><span class="chip">{p.sector}</span>'
        f'<span class="chip">{p.weight_percent}% of portfolio</span></div>'
        f'<div class="meta">{p.quantity:g} shares &middot; bought INR {p.buy_price:,.2f} '
        f'&middot; now INR {p.current_price:,.2f} &middot; '
        f'<b style="color:{"#2FBF71" if p.gain_percent >= 0 else "#E5484D"}">'
        f'{p.gain_percent:+.2f}% ({money(p.gain_absolute)})</b></div></div>',
        unsafe_allow_html=True)

with st.expander("Raw analysis tool output (what the model was given)"):
    st.code(st.session_state.raw, language="text")

# ----------------------------------------------------------------- chat
st.write("")
st.markdown("#### Ask about your statement")
st.caption("Answers come from your uploaded statement and live market data. "
           "The assistant is told never to calculate a figure itself.")

cols = st.columns(4)
suggestions = [
    "Which holding is my biggest risk?",
    "How is my IT sector doing?",
    "What is TCS trading at today?",
    "Which holdings am I losing money on?",
]
clicked = None
for col, s in zip(cols, suggestions):
    if col.button(s, use_container_width=True):
        clicked = s

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

typed = st.chat_input("Ask a follow-up question about your portfolio")
question = clicked or typed

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking your statement and live prices..."):
            answer = ask(st.session_state.agent, question)
        st.markdown(answer)
    st.session_state.history.append(("assistant", answer))

st.markdown('<div class="disc">This tool is informational only and is not personalised '
            'financial advice. Figures marked "live" come from a public market data feed and '
            'may be delayed. Figures marked "statement" come from the file you uploaded.</div>',
            unsafe_allow_html=True)
