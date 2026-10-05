"""Central configuration. All tunables live here so the rest of the code stays clean."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# --- LLM (Groq) ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
# GPT-OSS 120B is free on Groq and supports native tool-calling.
LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# --- Embeddings (fastembed / ONNX) ---
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # 384-dim, ONNX
EMBED_DIM: int = 384

# --- Retrieval ---
CHUNK_SIZE: int = 900          # characters per chunk
CHUNK_OVERLAP: int = 150       # character overlap between chunks
TOP_K: int = 4                 # chunks returned per search

# --- Agent ---
MAX_AGENT_STEPS: int = 5       # safety cap on the tool-calling loop

# --- Paths ---
DATA_DIR: str = os.getenv("DATA_DIR", "data")


def require_api_key() -> str:
    """Fail loudly with a helpful message if the key is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and add it to a .env file (see .env.example)."
        )
    return GROQ_API_KEY
