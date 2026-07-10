"""The agent loop: a transparent, framework-free tool-calling loop over Groq.

Returns both the final answer and a trace of the tools used, so the UI can show
*how* the agent reasoned — useful in interviews and for debugging.

Robustness note: open models (e.g. Llama 3.3 on Groq) occasionally emit a tool
call in a malformed text format like `<function=search_papers{"query": "x"}</function>`
instead of the structured `tool_calls` field. Groq rejects that with a 400
`tool_use_failed` error. Rather than crash, we parse the intended call out of the
error payload and continue — so the demo stays reliable for whoever runs it.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field

from groq import BadRequestError, Groq

import config
from agent import tools
from agent.prompts import AGENT_SYSTEM
from rag.store import VectorStore

# Matches the malformed tool call Groq/Llama sometimes emits in `failed_generation`.
_MALFORMED_CALL = re.compile(r"<function=(\w+)\s*(\{.*?\})\s*(?:</function>)?", re.DOTALL)


@dataclass
class AgentResult:
    answer: str
    tool_trace: list[str] = field(default_factory=list)   # human-readable steps
    context: list[str] = field(default_factory=list)      # passages the answer used


def _client() -> Groq:
    return Groq(api_key=config.require_api_key())


def _parse_malformed(failed_generation: str) -> list[dict]:
    """Recover intended tool calls from a malformed generation string."""
    calls: list[dict] = []
    for name, raw_args in _MALFORMED_CALL.findall(failed_generation or ""):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            continue
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )
    return calls


def _complete_with_tools(client: Groq, messages: list[dict]):
    """One turn of the loop. Returns (assistant_tool_calls, assistant_text).

    Falls back to parsing a malformed generation if Groq raises tool_use_failed.
    """
    try:
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            tools=tools.TOOL_SPECS,
            tool_choice="auto",
            temperature=config.LLM_TEMPERATURE,
        )
        msg = resp.choices[0].message
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (msg.tool_calls or [])
        ]
        return tool_calls, (msg.content or "")
    except BadRequestError as exc:
        failed = _extract_failed_generation(exc)
        recovered = _parse_malformed(failed)
        if recovered:
            return recovered, ""
        raise


def _extract_failed_generation(exc: BadRequestError) -> str:
    """Dig the `failed_generation` string out of a Groq 400 error."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", {})
        if isinstance(err, dict) and err.get("code") == "tool_use_failed":
            return err.get("failed_generation", "")
    return ""


def run_agent(question: str, store: VectorStore) -> AgentResult:
    """Answer a question, letting the model call tools until it's ready to respond."""
    client = _client()
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": question},
    ]
    trace: list[str] = []

    for _ in range(config.MAX_AGENT_STEPS):
        tool_calls, text = _complete_with_tools(client, messages)

        # No tool calls -> the model has produced its final answer.
        if not tool_calls:
            return AgentResult(answer=text, tool_trace=trace, context=list(tools.LAST_CONTEXT))

        # Record the assistant turn (with its tool calls) before appending results.
        messages.append({"role": "assistant", "content": text, "tool_calls": tool_calls})

        # Execute each requested tool and feed results back.
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            result = tools.dispatch(name, args, store)
            trace.append(f"{name}({args.get('query', '')!r})")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": result,
                }
            )

    # Hit the step cap — ask for a final answer with what we have.
    messages.append(
        {"role": "user", "content": "Give your best final answer now using what you found."}
    )
    final = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=config.LLM_TEMPERATURE,
    )
    return AgentResult(
        answer=final.choices[0].message.content or "",
        tool_trace=trace,
        context=list(tools.LAST_CONTEXT),
    )
