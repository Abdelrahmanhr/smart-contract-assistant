"""
guardrails/guardrails.py

Two-layer guardrail system:

  Layer 1 — Safety check
    Detects unsafe / off-topic user queries using keyword matching and an
    optional LLM-based intent classifier.

  Layer 2 — Grounding check
    Verifies that the LLM answer is grounded in the retrieved source chunks
    by computing embedding-based semantic similarity.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from app.config import (
    GUARDRAIL_SIMILARITY_THRESHOLD,
    UNSAFE_KEYWORDS,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)
from app.llm.prompts import SAFETY_CHECK_PROMPT

logger = logging.getLogger(__name__)


# ─── Layer 1: Safety / Intent ─────────────────────────────────────────────────

def check_safety_keywords(question: str) -> Tuple[bool, str]:
    """
    Fast keyword-based safety check.

    Returns:
        (is_safe, reason) — is_safe=False means the query is blocked.
    """
    lower = question.lower()
    for kw in UNSAFE_KEYWORDS:
        if kw in lower:
            reason = f"Query contains potentially unsafe keyword: '{kw}'"
            logger.warning("Safety block: %s", reason)
            return False, reason
    return True, ""


def check_safety_llm(
    question: str,
    llm: Optional[ChatOllama] = None,
) -> Tuple[bool, str]:
    """
    LLM-based safety classification.
    Falls back to True (safe) if the LLM is unreachable.

    Returns:
        (is_safe, reason)
    """
    _llm = llm or ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        num_predict=5,
    )
    chain = SAFETY_CHECK_PROMPT | _llm | StrOutputParser()
    try:
        result = chain.invoke({"question": question}).strip().upper()
        if result.startswith("UNSAFE"):
            return False, "LLM classified the query as unsafe."
        return True, ""
    except Exception as exc:
        logger.warning("LLM safety check failed (fallback to safe): %s", exc)
        return True, ""


def is_safe(
    question: str,
    use_llm: bool = False,
    llm: Optional[ChatOllama] = None,
) -> Tuple[bool, str]:
    """
    Combined safety gate.

    Args:
        question:  User query.
        use_llm:   Whether to also run the LLM safety classifier.
        llm:       Optional pre-built LLM instance.

    Returns:
        (is_safe, reason)
    """
    safe, reason = check_safety_keywords(question)
    if not safe:
        return False, reason

    if use_llm:
        safe, reason = check_safety_llm(question, llm=llm)

    return safe, reason


# ─── Layer 2: Grounding check ─────────────────────────────────────────────────

def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def check_grounding(
    answer: str,
    source_docs: List[Document],
    threshold: float = GUARDRAIL_SIMILARITY_THRESHOLD,
) -> Tuple[bool, float]:
    """
    Check whether the LLM answer is semantically grounded in the source chunks.

    Uses the same local SentenceTransformer embeddings as the vector store
    so no extra network calls are needed.

    Args:
        answer:      The LLM-generated answer.
        source_docs: Retrieved source documents.
        threshold:   Minimum cosine similarity to consider grounded.

    Returns:
        (is_grounded, max_similarity_score)
    """
    if not source_docs:
        logger.warning("Grounding check: no source docs provided.")
        return False, 0.0

    try:
        from sentence_transformers import SentenceTransformer
        from app.config import EMBEDDING_MODEL

        model = SentenceTransformer(EMBEDDING_MODEL)
        combined_context = " ".join(d.page_content for d in source_docs)
        embeddings = model.encode([answer, combined_context], normalize_embeddings=True)
        score = float(_cosine_similarity(embeddings[0].tolist(), embeddings[1].tolist()))

        is_grounded = score >= threshold
        logger.debug(
            "Grounding check: score=%.3f, threshold=%.3f → %s",
            score,
            threshold,
            "GROUNDED" if is_grounded else "NOT GROUNDED",
        )
        return is_grounded, score

    except Exception as exc:
        logger.warning("Grounding check failed (skipping): %s", exc)
        return True, 1.0  # fail-open: don't block on error


def apply_guardrails(
    question: str,
    answer: str,
    source_docs: List[Document],
    use_llm_safety: bool = False,
    llm: Optional[ChatOllama] = None,
) -> Tuple[str, dict]:
    """
    Apply both safety and grounding guardrails to a completed Q&A turn.

    Args:
        question:        The user's question.
        answer:          The LLM's answer.
        source_docs:     Retrieved source documents.
        use_llm_safety:  Whether to use LLM-based safety check.
        llm:             Optional LLM instance.

    Returns:
        (final_answer, guardrail_info_dict)
        final_answer may be replaced with a warning message if checks fail.
    """
    info: dict = {
        "safety_passed": True,
        "safety_reason": "",
        "grounding_passed": True,
        "grounding_score": 1.0,
    }

    # Safety check
    safe, reason = is_safe(question, use_llm=use_llm_safety, llm=llm)
    info["safety_passed"] = safe
    info["safety_reason"] = reason
    if not safe:
        return (
            f"⚠️ I'm unable to answer this question. {reason}",
            info,
        )

    # Grounding check
    grounded, score = check_grounding(answer, source_docs)
    info["grounding_passed"] = grounded
    info["grounding_score"] = score
    if not grounded:
        logger.warning("Answer may not be grounded (score=%.3f).", score)
        # Warn but don't block — append a disclaimer instead
        answer = (
            answer
            + "\n\n⚠️ *Note: This answer may contain information not directly "
            "supported by the uploaded document. Please verify.*"
        )

    return answer, info
