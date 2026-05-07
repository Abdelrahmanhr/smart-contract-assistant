"""
evaluation/evaluator.py

Lightweight evaluation pipeline measuring three RAG quality dimensions:

  1. Answer Relevance  — how relevant is the answer to the question?
  2. Faithfulness      — is the answer grounded in the retrieved context?
  3. Context Recall    — do the retrieved chunks actually contain the answer?

All metrics use local SentenceTransformer embeddings (cosine similarity)
so no external API is required.

Usage (standalone):
    python -m app.evaluation.evaluator
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """One evaluation example: question + expected answer + context chunks."""
    question: str
    expected_answer: str
    context_chunks: List[str]   # retrieved chunk texts
    generated_answer: str


@dataclass
class EvalResult:
    question: str
    answer_relevance: float     # 0–1
    faithfulness: float         # 0–1
    context_recall: float       # 0–1
    mean_score: float           # average of the three

    def __str__(self) -> str:
        return (
            f"Q: {self.question[:80]}\n"
            f"  Answer Relevance : {self.answer_relevance:.3f}\n"
            f"  Faithfulness     : {self.faithfulness:.3f}\n"
            f"  Context Recall   : {self.context_recall:.3f}\n"
            f"  Mean Score       : {self.mean_score:.3f}"
        )


# ─── Embedding helper ─────────────────────────────────────────────────────────

_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        from app.config import EMBEDDING_MODEL
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _embed(texts: List[str]) -> "np.ndarray":
    model = _get_embedding_model()
    return model.encode(texts, normalize_embeddings=True)


def _cosine(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b))   # normalized vectors: dot == cosine


# ─── Individual metrics ───────────────────────────────────────────────────────

def answer_relevance(question: str, generated_answer: str) -> float:
    """
    Measures how semantically relevant the answer is to the question.
    High score = the answer actually addresses the question.
    """
    vecs = _embed([question, generated_answer])
    return _cosine(vecs[0], vecs[1])


def faithfulness(generated_answer: str, context_chunks: List[str]) -> float:
    """
    Measures how grounded the answer is in the provided context.
    High score = answer content matches the context.
    """
    if not context_chunks:
        return 0.0
    combined_context = " ".join(context_chunks)
    vecs = _embed([generated_answer, combined_context])
    return _cosine(vecs[0], vecs[1])


def context_recall(expected_answer: str, context_chunks: List[str]) -> float:
    """
    Measures whether the retrieved chunks contain the expected answer.
    High score = the context covers the ground-truth answer.
    """
    if not context_chunks:
        return 0.0
    combined_context = " ".join(context_chunks)
    vecs = _embed([expected_answer, combined_context])
    return _cosine(vecs[0], vecs[1])


# ─── Main evaluator ───────────────────────────────────────────────────────────

def evaluate_sample(sample: EvalSample) -> EvalResult:
    """Evaluate a single Q&A sample and return all metrics."""
    ar = answer_relevance(sample.question, sample.generated_answer)
    ff = faithfulness(sample.generated_answer, sample.context_chunks)
    cr = context_recall(sample.expected_answer, sample.context_chunks)
    mean = (ar + ff + cr) / 3

    return EvalResult(
        question=sample.question,
        answer_relevance=round(ar, 4),
        faithfulness=round(ff, 4),
        context_recall=round(cr, 4),
        mean_score=round(mean, 4),
    )


def evaluate_dataset(samples: List[EvalSample]) -> Dict:
    """
    Evaluate a list of samples and return per-sample results plus averages.

    Returns:
        {
            "results": [EvalResult, ...],
            "averages": {"answer_relevance": ..., "faithfulness": ...,
                         "context_recall": ..., "mean_score": ...}
        }
    """
    results = [evaluate_sample(s) for s in samples]

    averages = {
        "answer_relevance": round(sum(r.answer_relevance for r in results) / len(results), 4),
        "faithfulness":     round(sum(r.faithfulness for r in results) / len(results), 4),
        "context_recall":   round(sum(r.context_recall for r in results) / len(results), 4),
        "mean_score":       round(sum(r.mean_score for r in results) / len(results), 4),
    }

    return {"results": results, "averages": averages}


def print_evaluation_report(eval_output: Dict) -> None:
    """Pretty-print the evaluation report to stdout."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    for result in eval_output["results"]:
        print(result)
        print()
    print("-" * 60)
    avgs = eval_output["averages"]
    print("AVERAGES")
    print(f"  Answer Relevance : {avgs['answer_relevance']:.3f}")
    print(f"  Faithfulness     : {avgs['faithfulness']:.3f}")
    print(f"  Context Recall   : {avgs['context_recall']:.3f}")
    print(f"  Mean Score       : {avgs['mean_score']:.3f}")
    print("=" * 60 + "\n")


def save_evaluation_report(eval_output: Dict, output_path: str | Path) -> None:
    """Save the evaluation report as a JSON file."""
    serialisable = {
        "results": [asdict(r) for r in eval_output["results"]],
        "averages": eval_output["averages"],
    }
    Path(output_path).write_text(json.dumps(serialisable, indent=2))
    logger.info("Evaluation report saved to %s", output_path)


# ─── Demo / self-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running built-in evaluation demo …")

    demo_samples = [
        EvalSample(
            question="What are the payment terms?",
            expected_answer="Payment is due within 30 days of invoice.",
            context_chunks=[
                "Section 4.2: All invoices are payable within 30 days of the invoice date.",
                "Late payments will incur a 2% monthly interest charge.",
            ],
            generated_answer="According to Section 4.2, payment is due within 30 days of the invoice date.",
        ),
        EvalSample(
            question="Who are the parties to the agreement?",
            expected_answer="Acme Corp and Beta Ltd.",
            context_chunks=[
                "This Agreement is entered into between Acme Corp ('Vendor') and Beta Ltd ('Client').",
            ],
            generated_answer="The parties are Acme Corp (Vendor) and Beta Ltd (Client).",
        ),
        EvalSample(
            question="What is the governing law?",
            expected_answer="The agreement is governed by the laws of New York.",
            context_chunks=[
                "Section 12: Termination clause applies after 30 days notice.",
                "Section 11: All payments must be made in USD.",
            ],
            generated_answer="I could not find specific information about governing law in the document.",
        ),
    ]

    output = evaluate_dataset(demo_samples)
    print_evaluation_report(output)
