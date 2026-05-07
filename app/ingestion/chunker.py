"""
ingestion/chunker.py

Splits LangChain Documents into smaller, overlapping chunks suitable for embedding.
Uses LangChain's RecursiveCharacterTextSplitter with configurable parameters.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# Separators are tried in order — largest logical unit first
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks.

    Each chunk inherits the original document's metadata and gains two
    additional fields:
        - chunk_index: zero-based position within the source document
        - total_chunks: total number of chunks produced from that document

    Args:
        documents:    Raw documents returned by document_loader.
        chunk_size:   Target character length of each chunk.
        chunk_overlap: Number of characters shared between adjacent chunks.

    Returns:
        A flat list of chunked Document objects.
    """
    if not documents:
        logger.warning("chunk_documents received an empty document list.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: List[Document] = []

    for doc in documents:
        raw_chunks = splitter.split_documents([doc])

        for idx, chunk in enumerate(raw_chunks):
            # Enrich metadata
            chunk.metadata.update(
                {
                    "chunk_index": idx,
                    "total_chunks": len(raw_chunks),
                }
            )
            all_chunks.append(chunk)

    logger.info(
        "Chunked %d document(s) → %d chunk(s) "
        "(chunk_size=%d, overlap=%d)",
        len(documents),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks
