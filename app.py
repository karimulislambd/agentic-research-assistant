"""Streamlit UI: upload papers, chat with the agent, and watch it self-evaluate.

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

import config
from agent.core import run_agent
from evaluation.judge import evaluate
from rag.ingest import ingest_pdfs
from rag.store import VectorStore

st.set_page_config(page_title="Agentic Research Assistant", page_icon="📚", layout="wide")


# ---- session state ----
if "store" not in st.session_state:
    st.session_state.store = VectorStore()
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: question, answer, trace, eval


def _sidebar() -> None:
    with st.sidebar:
        st.header("📚 Your papers")
        uploaded = st.file_uploader(
            "Upload PDF papers", type="pdf", accept_multiple_files=True
        )
        if uploaded and st.button("Ingest", type="primary", use_container_width=True):
            files = [(f.name, f.read()) for f in uploaded]
            with st.spinner("Embedding papers (ONNX)…"):
                n = ingest_pdfs(files, st.session_state.store)
            st.success(f"Added {n} chunks from {len(files)} file(s).")

        store: VectorStore = st.session_state.store
        if not store.is_empty:
            st.caption("Indexed sources:")
            for s in store.sources:
                st.write(f"• {s}")

        st.divider()
        if not config.GROQ_API_KEY:
            st.error("GROQ_API_KEY missing — set it in your .env file.")
        else:
            st.caption(f"LLM: `{config.LLM_MODEL}`  ·  Embed: `{config.EMBED_MODEL}`")


def _score_badge(label: str, value: int) -> str:
    color = "#16a34a" if value >= 4 else "#d97706" if value >= 3 else "#dc2626"
    return (
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:10px;font-size:0.8rem'>{label}: {value}/5</span>"
    )


def main() -> None:
    st.title("📚 Agentic Research Assistant")
    st.caption(
        "Ask questions across your uploaded papers. An agent retrieves, cites, and "
        "**scores its own answers** for faithfulness & relevance."
    )
    _sidebar()

    store: VectorStore = st.session_state.store

    # Render history.
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            if turn["trace"]:
                st.caption("🛠 Tools used: " + "  →  ".join(turn["trace"]))
            ev = turn["eval"]
            if ev:
                st.markdown(
                    _score_badge("Faithfulness", ev.faithfulness)
                    + " &nbsp; "
                    + _score_badge("Relevance", ev.relevance),
                    unsafe_allow_html=True,
                )
                st.caption(f"⚖️ Judge: {ev.reason}")

    question = st.chat_input("Ask about your papers…")
    if not question:
        return

    if store.is_empty:
        st.warning("Upload and ingest at least one PDF first (see the sidebar).")
        return
    if not config.GROQ_API_KEY:
        st.error("Set GROQ_API_KEY in .env to run the agent.")
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Agent is researching…"):
            result = run_agent(question, store)
        st.markdown(result.answer)
        if result.tool_trace:
            st.caption("🛠 Tools used: " + "  →  ".join(result.tool_trace))
        with st.spinner("Evaluating answer…"):
            ev = evaluate(question, result.context, result.answer)
        st.markdown(
            _score_badge("Faithfulness", ev.faithfulness)
            + " &nbsp; "
            + _score_badge("Relevance", ev.relevance),
            unsafe_allow_html=True,
        )
        st.caption(f"⚖️ Judge: {ev.reason}")

    st.session_state.history.append(
        {
            "question": question,
            "answer": result.answer,
            "trace": result.tool_trace,
            "eval": ev,
        }
    )


if __name__ == "__main__":
    main()
