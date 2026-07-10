"""Unit tests for the retrieval layer. These run in CI without any API key.

They cover the parts that don't need Groq: chunking and vector search.
"""
from __future__ import annotations

from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from rag.ingest import _chunk_page, pdf_to_chunks
from rag.store import Chunk, VectorStore


def _make_pdf(lines: list[str]) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 750
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()
    return buf.getvalue()


def test_chunk_page_respects_size():
    text = "word " * 500  # ~2500 chars
    chunks = _chunk_page(text)
    assert len(chunks) > 1
    assert all(len(c) <= 900 for c in chunks)


def test_chunk_page_empty():
    assert _chunk_page("   \n\n  ") == []


def test_pdf_to_chunks_sets_metadata():
    pdf = _make_pdf(["Transformers use self-attention.", "BERT is bidirectional."])
    chunks = pdf_to_chunks(pdf, source="paper.pdf", start_id=0)
    assert chunks, "expected at least one chunk"
    assert chunks[0].source == "paper.pdf"
    assert chunks[0].page == 1
    assert chunks[0].chunk_id == 0


def test_vector_search_returns_relevant_chunk():
    store = VectorStore()
    store.add(
        [
            Chunk("Self-attention lets tokens attend to each other.", "a.pdf", 1, 0),
            Chunk("Bananas are a good source of potassium.", "b.pdf", 1, 1),
        ]
    )
    hits = store.search("how does attention work", top_k=1)
    assert hits
    top_chunk, score = hits[0]
    assert top_chunk.source == "a.pdf"
    assert 0.0 <= score <= 1.0001
