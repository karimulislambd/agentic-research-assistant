# Agentic Research Assistant — with self-evaluation

[![CI](https://github.com/karimulislambd/agentic-research-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/karimulislambd/agentic-research-assistant/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://agentic-research-assistant-karimulislambd.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-000000?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)

> An LLM **agent** that answers questions across your uploaded research papers, cites its
> sources, and **scores its own answers** for faithfulness and relevance.

Most "chat with your PDF" demos stop at retrieval. This one adds an **evaluation layer**:
every answer is judged (1–5) on whether it is *grounded in the retrieved text* and whether
it *actually answers the question* — turning a chatbot into a measurable, benchmarkable system.

**Live demo:** https://agentic-research-assistant-karimulislambd.streamlit.app/

---

## Why this project

| Skill it demonstrates | Where |
|---|---|
| **Agentic tool-use** | Framework-free tool-calling loop (`agent/core.py`) |
| **RAG** | ONNX embeddings + FAISS vector search (`rag/`) |
| **LLM evaluation** | Faithfulness/relevance LLM-as-judge (`evaluation/judge.py`) |
| **Prompt engineering** | Versioned system prompts (`agent/prompts.py`) |
| **MLOps** | Dockerfile, GitHub Actions CI, unit tests |

## Architecture

```
             ┌───────────────────────────────────────────┐
 question ─► │  Agent loop (Groq · Llama 3.3 · tool-call) │
             │    ├─ search_papers → FAISS over your PDFs │
             │    └─ web_search    → background context    │
             └───────────────┬───────────────────────────┘
                             │  answer + citations
                             ▼
             ┌───────────────────────────────────────────┐
             │  LLM-as-judge → faithfulness & relevance   │  ◄─ the differentiator
             └───────────────────────────────────────────┘
```

## Tech stack

- **LLM:** Groq (`llama-3.3-70b-versatile`) — free tier, native tool-calling
- **Embeddings:** `fastembed` (`bge-small-en-v1.5`, **ONNX**) — lightweight, no PyTorch
- **Vector search:** FAISS (`IndexFlatIP`, cosine)
- **UI:** Streamlit
- **Quality:** pytest · ruff · GitHub Actions · Docker

## Quickstart

```bash
git clone <your-repo-url>
cd ai-research-assistant

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then paste your free Groq key
streamlit run app.py
```

Get a **free** Groq API key at <https://console.groq.com>.

## Run the tests

```bash
pytest -q          # retrieval + chunking tests, no API key needed
ruff check .
```

## How it works (30-second tour)

1. **Ingest** — PDFs are parsed (`pypdf`), chunked with overlap, embedded with an ONNX
   model, and indexed in FAISS. Each chunk keeps its `source` + `page` so answers can cite it.
2. **Agent** — the model is given two tools and decides when to search the papers, search the
   web, or answer. The loop is written by hand (no LangChain) so the reasoning is transparent.
3. **Evaluate** — a second LLM call judges the answer against the exact passages retrieved,
   returning faithfulness and relevance scores shown live in the UI.

## Roadmap

- [ ] Persist the FAISS index to disk between sessions
- [ ] Batch-evaluation mode over a question set → aggregate quality dashboard
- [ ] Answer caching + cost/latency metrics per query

---

Built by **Md Karimul Islam** — AI/ML Engineer · Computer Vision · LLM · XAI.
