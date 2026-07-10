"""End-to-end smoke test against the live Groq API.

Builds a tiny paper corpus, runs the agent, prints the answer + citations +
tool trace, then runs the LLM-as-judge. Proves the full pipeline works before
we deploy. Run:  python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

# Make the project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

from agent.core import run_agent  # noqa: E402
from evaluation.judge import evaluate  # noqa: E402
from rag.ingest import ingest_pdfs  # noqa: E402
from rag.store import VectorStore  # noqa: E402

PAPER_TEXT = [
    "MEFNet: A Hybrid CNN for Waste Classification.",
    "Abstract. We propose MEFNet, a fusion of MobileNetV3 and EfficientNetB0",
    "for classifying municipal waste into 12 categories. On our benchmark,",
    "MEFNet reaches 97.71 percent accuracy, outperforming ResNet50V2,",
    "EfficientNetV2, and ConvNeXt-Tiny. We apply Grad-CAM to make the model's",
    "decisions interpretable, highlighting the regions that drive each prediction.",
    "The fusion architecture keeps inference lightweight enough for edge devices.",
]


def main() -> None:
    store = VectorStore()

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 750
    for line in PAPER_TEXT:
        c.drawString(72, y, line)
        y -= 22
    c.save()

    added = ingest_pdfs([("MEFNet.pdf", buf.getvalue())], store)
    print(f"[ingest] {added} chunks from 1 paper\n")

    question = "What accuracy does MEFNet achieve and which models does it outperform?"
    print(f"[question] {question}\n")

    result = run_agent(question, store)
    print("[tools used]", " -> ".join(result.tool_trace) or "(none)")
    print("\n[answer]\n" + result.answer + "\n")

    ev = evaluate(question, result.context, result.answer)
    print(f"[judge] faithfulness={ev.faithfulness}/5  relevance={ev.relevance}/5")
    print(f"[judge] reason: {ev.reason}")

    ok = bool(result.answer) and ev.faithfulness >= 3 and ev.relevance >= 3
    print("\n" + ("PASS - pipeline is live and grounded" if ok else "CHECK - review output above"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
