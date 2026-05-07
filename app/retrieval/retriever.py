"""
retrieval/retriever.py

Wraps the Chroma vector store in a LangChain retriever.
Supports score-threshold filtering to avoid returning irrelevant chunks.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD

logger = logging.getLogger(__name__)


def get_retriever(
    vector_store: Chroma,
    top_k: int = RETRIEVAL_TOP_K,
    score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
) -> BaseRetriever:
    """
    Return a LangChain retriever backed by the given Chroma vector store.

    Uses similarity-score thresholding so that semantically irrelevant chunks
    are not passed to the LLM.

    Args:
        vector_store:     A loaded or freshly built Chroma instance.
        top_k:            Maximum number of chunks to return.
        score_threshold:  Minimum cosine-similarity score (0–1).

    Returns:
        A LangChain BaseRetriever ready for use in a chain.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": top_k,
            "score_threshold": score_threshold,
        },
    )
    logger.debug(
        "Retriever created: top_k=%d, score_threshold=%.2f", top_k, score_threshold
    )
    return retriever


def retrieve_chunks(
    vector_store: Chroma,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
) -> List[Tuple[Document, float]]:
    """
    Run a raw similarity search and return (Document, score) pairs.
    Useful for the evaluation pipeline or when you need the actual scores.

    Args:
        vector_store:     Chroma instance.
        query:            Natural-language query string.
        top_k:            Maximum number of results.
        score_threshold:  Minimum similarity score.

    Returns:
        List of (Document, similarity_score) tuples, sorted by score desc.
    """
    results: List[Tuple[Document, float]] = (
        vector_store.similarity_search_with_relevance_scores(query, k=top_k)
    )

    filtered = [
        (doc, score) for doc, score in results if score >= score_threshold
    ]

    logger.debug(
        "Query='%s' → %d/%d chunks passed threshold %.2f",
        query[:80],
        len(filtered),
        len(results),
        score_threshold,
    )
    return filtered
