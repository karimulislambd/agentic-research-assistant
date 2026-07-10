"""The agent loop: a transparent, framework-free tool-calling loop over Groq.

Returns both the final answer and a trace of the tools used, so the UI can show
*how* the agent reasoned — useful in interviews and for debugging.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from groq import Groq

import config
from agent import tools
from agent.prompts import AGENT_SYSTEM
from rag.store import VectorStore


@dataclass
class AgentResult:
    answer: str
    tool_trace: list[str] = field(default_factory=list)   # human-readable steps
    context: list[str] = field(default_factory=list)      # passages the answer used


def _client() -> Groq:
    return Groq(api_key=config.require_api_key())


def run_agent(question: str, store: VectorStore) -> AgentResult:
    """Answer a question, letting the model call tools until it's ready to respond."""
    client = _client()
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": question},
    ]
    trace: list[str] = []

    for _ in range(config.MAX_AGENT_STEPS):
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            tools=tools.TOOL_SPECS,
            tool_choice="auto",
            temperature=config.LLM_TEMPERATURE,
        )
        msg = resp.choices[0].message

        # No tool calls -> the model has produced its final answer.
        if not msg.tool_calls:
            return AgentResult(
                answer=msg.content or "",
                tool_trace=trace,
                context=list(tools.LAST_CONTEXT),
            )

        # Record the assistant turn (with its tool calls) before appending results.
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        # Execute each requested tool and feed results back.
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = tools.dispatch(tc.function.name, args, store)
            preview = args.get("query", "")
            trace.append(f"{tc.function.name}({preview!r})")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
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
