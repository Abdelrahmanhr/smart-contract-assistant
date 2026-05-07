"""
summarization/summarizer.py

Produces a structured summary of the entire uploaded document using
LangChain's map-reduce summarisation pattern:

  Map   — Each chunk is summarised independently.
  Reduce — All chunk summaries are combined into one final summary.

This avoids context-window overflow for large documents.
"""

from __future__ import annotations

import logging
from typing import List

from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from app.config import (
    CHUNK_OVERLAP,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SUMMARIZATION_CHUNK_SIZE,
    SUMMARIZATION_MAX_TOKENS,
)
from app.llm.chain import get_llm

logger = logging.getLogger(__name__)

# ─── Prompts ──────────────────────────────────────────────────────────────────

_MAP_TEMPLATE = """You are a legal-document analyst.
Summarise the following section of a contract or legal document.
Focus on: parties involved, key obligations, important dates, and monetary values.
Be concise.

SECTION:
{text}

SECTION SUMMARY:"""

_REDUCE_TEMPLATE = """You are a legal-document analyst.
Below are summaries of different sections of a legal document.
Combine them into a single, coherent executive summary covering:
1. Parties and their roles
2. Purpose / subject of the agreement
3. Key obligations and rights
4. Important dates and deadlines
5. Financial terms
6. Termination and dispute resolution clauses (if present)

SECTION SUMMARIES:
{text}

FINAL EXECUTIVE SUMMARY:"""

MAP_PROMPT = PromptTemplate(input_variables=["text"], template=_MAP_TEMPLATE)
REDUCE_PROMPT = PromptTemplate(input_variables=["text"], template=_REDUCE_TEMPLATE)


# ─── Summarizer ───────────────────────────────────────────────────────────────

def summarize_document(
    documents: List[Document],
    llm: ChatOllama | None = None,
) -> str:
    """
    Produce an executive summary for a list of Documents.

    For short documents (≤ 1 chunk), uses the "stuff" strategy.
    For longer documents, uses "map_reduce" to avoid context overflow.

    Args:
        documents: Raw or chunked Document objects.
        llm:       Optional pre-built LLM instance.

    Returns:
        A plain-text executive summary string.
    """
    if not documents:
        return "No document content available to summarise."

    _llm = llm or get_llm(max_tokens=SUMMARIZATION_MAX_TOKENS)

    # Re-chunk with larger chunks for summarization (less fragmentation)
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SUMMARIZATION_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    strategy = "stuff" if len(chunks) <= 3 else "map_reduce"
    logger.info(
        "Summarising %d chunk(s) using '%s' strategy.", len(chunks), strategy
    )

    if strategy == "stuff":
        chain = load_summarize_chain(
            llm=_llm,
            chain_type="stuff",
            prompt=REDUCE_PROMPT,
        )
    else:
        chain = load_summarize_chain(
            llm=_llm,
            chain_type="map_reduce",
            map_prompt=MAP_PROMPT,
            combine_prompt=REDUCE_PROMPT,
            verbose=False,
        )

    try:
        result = chain.invoke({"input_documents": chunks})
        summary = result.get("output_text", "").strip()
        logger.info("Summary produced (%d chars).", len(summary))
        return summary
    except Exception as exc:
        logger.error("Summarisation failed: %s", exc)
        return f"Summarisation failed: {exc}"
