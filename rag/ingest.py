"""Turn uploaded PDFs into retrievable, citable chunks."""
from __future__ import annotations

import re
from io import BytesIO

from pypdf import PdfReader

import config
from rag.store import Chunk, VectorStore


def _clean(text: str) -> str:
    """Collapse the whitespace noise that PDF extraction leaves behind."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_page(text: str) -> list[str]:
    """Sliding-window chunking over a single page's text."""
    text = _clean(text)
    if not text:
        return []
    size, overlap = config.CHUNK_SIZE, config.CHUNK_OVERLAP
    step = max(size - overlap, 1)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def pdf_to_chunks(data: bytes, source: str, start_id: int = 0) -> list[Chunk]:
    """Parse one PDF (as bytes) into a list of Chunk objects."""
    reader = PdfReader(BytesIO(data))
    chunks: list[Chunk] = []
    cid = start_id
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        for piece in _chunk_page(page_text):
            chunks.append(Chunk(text=piece, source=source, page=page_num, chunk_id=cid))
            cid += 1
    return chunks


def ingest_pdfs(files: list[tuple[str, bytes]], store: VectorStore) -> int:
    """Ingest (name, bytes) pairs into the store. Returns number of chunks added."""
    added = 0
    for name, data in files:
        chunks = pdf_to_chunks(data, source=name, start_id=len(store.chunks) + added)
        store.add(chunks)
        added += len(chunks)
    return added
