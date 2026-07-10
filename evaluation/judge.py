"""LLM-as-judge evaluation of answer quality.

This is the project's differentiator: every answer is scored for *faithfulness*
(is it grounded in the retrieved context?) and *relevance* (does it answer the
question?). It turns a chatbot into a measurable, benchmarkable system.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from groq import Groq

import config
from agent.prompts import JUDGE_SYSTEM


@dataclass
class Evaluation:
    faithfulness: int   # 1..5
    relevance: int      # 1..5
    reason: str

    @property
    def ok(self) -> bool:
        return self.faithfulness >= 4 and self.relevance >= 4


def _parse(raw: str) -> Evaluation:
    """Pull the JSON verdict out of the model's reply, defensively."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return Evaluation(0, 0, "could not parse judge output")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Evaluation(0, 0, "invalid judge JSON")
    return Evaluation(
        faithfulness=int(data.get("faithfulness", 0)),
        relevance=int(data.get("relevance", 0)),
        reason=str(data.get("reason", "")),
    )


def evaluate(question: str, context: list[str], answer: str) -> Evaluation:
    """Score one answer. Returns neutral zeros if there is nothing to judge."""
    if not answer.strip():
        return Evaluation(0, 0, "empty answer")
    client = Groq(api_key=config.require_api_key())
    ctx = "\n\n".join(context) if context else "(no retrieved context)"
    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}\n\nANSWER:\n{answer}"
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    return _parse(resp.choices[0].message.content or "")
