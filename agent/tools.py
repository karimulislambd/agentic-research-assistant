"""Tool definitions + implementations the agent can call.

Each tool is a plain Python function plus a JSON schema Groq/OpenAI-style tool spec.
`web_search` is intentionally dependency-free (uses the free DuckDuckGo HTML endpoint)
so the demo needs no extra API keys; it degrades gracefully if offline.
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request

from rag.store import VectorStore

# The last search's retrieved chunks are stashed here so the evaluation layer
# can judge the answer against exactly what the agent saw.
LAST_CONTEXT: list[str] = []


def search_papers(store: VectorStore, query: str) -> str:
    """Retrieve the most relevant passages from the uploaded papers."""
    hits = store.search(query)
    LAST_CONTEXT.clear()
    if not hits:
        return "No relevant passages found in the uploaded papers."
    lines = []
    for chunk, score in hits:
        tag = f"[{chunk.source}, p.{chunk.page}]"
        LAST_CONTEXT.append(f"{tag} {chunk.text}")
        lines.append(f"{tag} (relevance {score:.2f})\n{chunk.text}")
    return "\n\n---\n\n".join(lines)


def web_search(query: str, max_results: int = 3) -> str:
    """Lightweight web search via DuckDuckGo's HTML endpoint (no API key needed)."""
    try:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:  # offline / blocked — fail soft
        return f"web_search unavailable ({exc})."

    snippets = re.findall(r'result__snippet"[^>]*>(.*?)</a>', html, flags=re.DOTALL)
    cleaned = [re.sub(r"<[^>]+>", "", s).strip() for s in snippets[:max_results]]
    cleaned = [c for c in cleaned if c]
    return "\n".join(f"- {c}" for c in cleaned) or "No web results found."


# ---- Tool specs advertised to the model ----
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search the user's uploaded research papers for passages "
            "relevant to a query. Returns passages with [source, p.PAGE] citations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused search query for the papers.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web for general background context "
            "that is not in the uploaded papers. Use sparingly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The web search query."}
                },
                "required": ["query"],
            },
        },
    },
]


def dispatch(name: str, args: dict, store: VectorStore) -> str:
    """Route a tool call from the model to its implementation."""
    if name == "search_papers":
        return search_papers(store, args.get("query", ""))
    if name == "web_search":
        return web_search(args.get("query", ""))
    return f"Unknown tool: {name}"
