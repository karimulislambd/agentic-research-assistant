"""A small, self-contained vector store: fastembed (ONNX) embeddings + FAISS index.

Kept deliberately minimal and framework-free so the retrieval logic is easy to read
in an interview — no LangChain magic hiding the moving parts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import faiss
import numpy as np
from fastembed import TextEmbedding

import config


@dataclass
class Chunk:
    """One retrievable passage plus enough metadata to cite it."""

    text: str
    source: str          # file name the chunk came from
    page: int            # 1-indexed page number
    chunk_id: int        # position within the whole corpus


@dataclass
class VectorStore:
    """Holds chunks + a FAISS index over their embeddings."""

    chunks: list[Chunk] = field(default_factory=list)
    _index: faiss.Index | None = None
    _embedder: TextEmbedding | None = None

    @property
    def embedder(self) -> TextEmbedding:
        # Lazy-load: the ONNX model downloads once, then is cached on disk.
        if self._embedder is None:
            self._embedder = TextEmbedding(model_name=config.EMBED_MODEL)
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self.embedder.embed(texts)), dtype="float32")
        # Normalize so inner-product == cosine similarity.
        faiss.normalize_L2(vecs)
        return vecs

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = self._embed([c.text for c in chunks])
        if self._index is None:
            self._index = faiss.IndexFlatIP(config.EMBED_DIM)
        self._index.add(vecs)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = config.TOP_K) -> list[tuple[Chunk, float]]:
        """Return the most relevant chunks with their similarity scores."""
        if self._index is None or not self.chunks:
            return []
        qv = self._embed([query])
        k = min(top_k, len(self.chunks))
        scores, idxs = self._index.search(qv, k)
        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], idxs[0], strict=False):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    @property
    def is_empty(self) -> bool:
        return self._index is None or not self.chunks

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})
